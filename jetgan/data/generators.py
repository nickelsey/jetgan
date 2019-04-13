""" 
Generators for simulated jets, and matching detector level jets
with varying levels of detector smearing. Used to generate 
training samples for detector model
"""

from __future__ import print_function
from __future__ import division

import fastjet
from numpy.random import normal as gauss_gen
from pythia8 import Pythia


def MakePseudoJet(p):
    pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
    pj.set_user_index(int(p.charge()))
    return pj


class PythiaGenerator:
    """
    Initialize the generator with settings for the pythia generator and
    as well as fastjet. Can set pT Hat range for pythia, jet radius, min jet 
    pt, and track level cuts (pt, eta)
    """

    def __init__(self, pt_hat_min=20.0, pt_hat_max=30.0, jet_radius=0.5,
                 track_pt_min=0.2, track_pt_max=30.0, track_eta_max=1.0,
                 jet_pt_min=10.0, gauss_smear = False, tpc_sim = False,
                 star_sim = False, gauss_mean=1.0, gauss_sigma=0.2):

        # generator settings
        self.pt_hat_min = pt_hat_min
        self.pt_hat_max = pt_hat_max
        self.gen_args = []

        # generator
        self.seed = 1
        self.gen = Pythia()
        self.initialized = False

        # fastjet settings
        self.jet_radius = jet_radius
        self.track_pt_min = track_pt_min
        self.track_pt_max = track_pt_max
        self.track_eta_max = track_eta_max
        self.jet_pt_min = jet_pt_min
        self.jet_eta_max = track_eta_max - jet_radius
        self.jet_alg = fastjet.antikt_algorithm

        # lists used to store truth and detector particles
        self.truth_particles = []
        self.detector_particles = []

        # lists used for generator and detector level jets
        self.gen_jet = []
        self.det_jet = []

        # smearing settings
        self.gauss_smear = gauss_smear
        self.tpc_sim = tpc_sim
        self.star_sim = star_sim

        # smearing parameters
        self.gauss_mean = gauss_mean
        self.gauss_sigma = gauss_sigma

        self.cluster_seq = []


    def add_pythia_command(self, arg):
        """
        allows the user to add custom arguments to the pythia generator.
        must be a valid Pythia setting string, which can be found in the
        Pythia documentation
        """
        self.gen_args.append(arg)

    def init(self):
        """
        must be called before next(). Initializes the generator, and creates
        all reused fastjet objects (selectors, jet definition). Returns the 
        status of the pythia generator
        """
        # first initialize Pythia
        self.gen.readString("Beams:eCM = 200.0")
        self.gen.readString("HardQCD:all = on")
        self.gen.readString("Random:setSeed = on")
        self.gen.readString('Random:seed = ' + str(self.seed))
        self.gen.readString('PhaseSpace:pTHatMin = ' + str(self.pt_hat_min))
        self.gen.readString('PhaseSpace:pTHatMax = ' + str(self.pt_hat_max))
        for setting in self.gen_args:
            try:
                self.gen.readString(setting)
            except:
                print('pythia read string failed')

        # create jet definition
        self.jet_def = fastjet.JetDefinition(self.jet_alg, self.jet_radius)

        # create fastjet selectors for tracks and jets
        self.track_selector = (fastjet.SelectorAbsEtaMax(self.track_eta_max) *
                               fastjet.SelectorPtRange(self.track_pt_min,
                                                       self.track_pt_max))
        self.jet_selector = (fastjet.SelectorAbsEtaMax(self.jet_eta_max) *
                             fastjet.SelectorPtMin(self.jet_pt_min))

        self.initialized = self.gen.init()

        if not self.star_sim and not self.gauss_smear and not self.tpc_sim:
            print('Warning: no detector simulation method has been chosen',
                  'disabling detector simulation')

        return self.initialized

    def smear_tracks(self):
        """
        creates the detector dataset using various methods depending on 
        user choice
        """
        self.detector_particles = []

        if self.gauss_smear:
            smearing = gauss_gen(loc=self.gauss_mean, scale=self.gauss_sigma, size=(len(self.truth_particles)))
            self.detector_particles = smearing * self.truth_particles
        elif self.tpc_sim:
            self.detector_particles = self.truth_particles
        elif self.star_sim:
            self.detector_particles = self.truth_particles

        # apply selector
        self.detector_particles = self.track_selector(self.detector_particles)
        
        return


    def next(self):
        """
        Calls Pythia to create the next event. If successful, clusters both 
        the truth level data and the smeared data, and returns true. Otherwise, 
        returns false
        """

        self.gen_jet = []
        self.det_jet = []
        self.truth_particles.clear()
        self.cluster_seq.clear()

        if not self.gen.next():
            return False
        
        # build our list of final state particles
        self.truth_particles = [MakePseudoJet(p) for p in self.gen.event if p.isFinal() and p.isVisible()]

        # build "detector level" tracks by smearing the truth particles
        # if the user has selected some sort of modification
        self.smear_tracks()

        # perform jetfinding for truth and detector jets 
        truth_cluster = fastjet.ClusterSequence(self.truth_particles, self.jet_def)
        det_cluster = fastjet.ClusterSequence(self.detector_particles, self.jet_def)

        self.gen_jet = fastjet.sorted_by_pt(self.jet_selector(truth_cluster.inclusive_jets()))
        self.det_jet = fastjet.sorted_by_pt(self.jet_selector(det_cluster.inclusive_jets()))

        self.cluster_seq.append(truth_cluster)
        self.cluster_seq.append(det_cluster)

        return

    def detector_jets(self):
        """
        Return detector level jets
        """
        return self.det_jet
    
    def generator_jets(self):
        """
        Return generator level jets
        """
        return self.gen_jet
