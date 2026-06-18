import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE, TF_WEIGHTS, TF_WEIGHTS_NOBONE

# a Gene Regulatory Network
class GRN:
    # 2D grid (left/right/up/down)
    diffusion_sites_qt = 4

    def __init__(self, promoter_threshold=0.95, max_voxels=27, cube_face_size=3,
                  genotype=None, voxel_types='withbone', env_conditions=None, plastic=None, use_diffusion=True): #diffusion added

        self.max_voxels = max_voxels
        self.genotype = genotype
        self.env_conditions = env_conditions
        self.plastic = plastic
        self.cells = []
        self.phenotype = None
        self.genes = []
        self.quantity_voxels = 0

        self.regulatory_transcription_factor_idx = 0
        self.regulatory_v1_idx = 1
        self.regulatory_v2_idx = 2
        self.transcription_factor_idx = 3
        self.transcription_factor_amount_idx = 4
        self.diffusion_site_idx = 5
        self.types_nucleotides = 6

        self.promoter_threshold = promoter_threshold
        self.concentration_decay = 0.005
        self.cube_face_size = cube_face_size

        # NOTE: if u increase number of reg tfs without increasing voxels tf or geno size,
        # too many tiny robots are sampled
        self.structural_products = None
        # number of regulatory tfs
        self.regulatory_products = 2

        if voxel_types == 'withbone':
            self.structural_products = VOXEL_TYPES
            self.tf_weights = TF_WEIGHTS
        if voxel_types == 'nobone':
            self.structural_products = VOXEL_TYPES_NOBONE
            self.tf_weights = TF_WEIGHTS_NOBONE

        # structural_tfs use initial indexes (excludes voxel_type offphase
        # and regulatory tfs uses final (leftover) indexes
        self.structural_tfs = []
        for tf in range(1, len(self.structural_products)):
            self.structural_tfs.append(f'TF{tf}')

        self.increase_scaling = 100
        self.intra_diffusion_rate = self.concentration_decay/2
        self.inter_diffusion_rate = self.intra_diffusion_rate/8
        self.dev_steps = 200
        self.concentration_threshold = np.minimum(0.1, self.genotype[0])

        #  params for phase alternation
        self.offphase_alternation_param = float(self.genotype[1])
        self.offphase_alternation_range = [1, 4]
        self._phase_run = 0

        self.genotype = self.genotype[2:]

        ############################
        # For the TF summary 
        self.tf_history = []
        ###########################

        ############################
        # For diffusion
        self.use_diffusion = use_diffusion
        ###########################

    def develop(self):
        k_min, k_max = self.offphase_alternation_range
        span = (k_max - k_min + 1)
        k = k_min + int(self.offphase_alternation_param * span)
        self.offphase_alternation_k = min(k, k_max)

        self.develop_body()

        #######################################################
        # Testing TF concentrations summary 
        print("Final TF concentrations:", self.tf_history[-1])

        # Checking TF concentrations over time
        #print("TF history:")
        #for step, tf_state in enumerate(self.tf_history[:10]):
        #    print(step, tf_state)
        #######################################################
        
        return self.phenotype

    def develop_body(self):
        self.gene_parser()
        self.regulate()

    def develop_knockout(self, knockouts):
        self.gene_parser()

        if knockouts is not None:
            self.genes = self.genes[np.logical_not(np.isin(np.arange(self.genes.shape[0]), knockouts))]

        self.regulate()
        return self.phenotype, self.genes

    # parses genotype to discover promoter sites and compose genes
    def gene_parser(self):
        nucleotide_idx = 0
        while nucleotide_idx < len(self.genotype):

            if self.genotype[nucleotide_idx] < self.promoter_threshold:
                # if there are nucleotides enough to compose a gene
                if (len(self.genotype)-1-nucleotide_idx) >= self.types_nucleotides:
                    regulatory_transcription_factor = self.genotype[nucleotide_idx+self.regulatory_transcription_factor_idx+1]
                    regulatory_v1 = self.genotype[nucleotide_idx+self.regulatory_v1_idx+1]
                    regulatory_v2 = self.genotype[nucleotide_idx+self.regulatory_v2_idx+1]
                    transcription_factor = self.genotype[nucleotide_idx+self.transcription_factor_idx+1] # gene product
                    transcription_factor_amount = max(0.1, self.genotype[nucleotide_idx+self.transcription_factor_amount_idx+1])
                    diffusion_site = self.genotype[nucleotide_idx+self.diffusion_site_idx+1]

                    # converts tfs values into labels
                    # remember that structural_tfs use initial indexes and regulatory tfs uses final (leftover) indexes
                    limits, total = self.build_tf_limits(self.structural_products,self.regulatory_products, self.tf_weights)
                    regulatory_transcription_factor_label = self.tf_value_to_label(regulatory_transcription_factor, limits)
                    transcription_factor_label = self.tf_value_to_label(transcription_factor, limits)

                    #  converts diffusion sites values into labels
                    range_size = 1.0 / GRN.diffusion_sites_qt
                    diffusion_site_label = min(int(diffusion_site / range_size), GRN.diffusion_sites_qt - 1)

                    gene = [regulatory_transcription_factor_label, regulatory_v1, regulatory_v2,
                            transcription_factor_label, transcription_factor_amount, diffusion_site_label]
               
                    self.genes.append(gene)

                    nucleotide_idx += self.types_nucleotides
            nucleotide_idx += 1
        self.genes = np.array(self.genes)

    def build_tf_limits(self, structural_products, regulatory_products, tf_weights):
        structural_products = dict(list(structural_products.items())[:-1])
        weights = []
        for name in structural_products:
            weights.append(float(tf_weights[name]))
        reg_w = float(tf_weights.get("regulatory", 1.0))
        weights.extend([reg_w] * regulatory_products)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        limits = np.concatenate(([0.0], np.cumsum(weights)))
        limits[-1] = 1.0  # numeric safety

        return limits, weights.size

    def tf_value_to_label(self, value, limits):
        # Clamp into [0,1) to avoid edge cases (value==1.0)
        v = float(value)
        if v >= 1.0:
            v = np.nextafter(1.0, 0.0)
        elif v < 0.0:
            v = 0.0
        idx = np.searchsorted(limits, v, side="right") - 1
        idx = max(0, min(idx, len(limits) - 2))
        return f"TF{idx + 1}"

    def net_parser(self):

        connections = []
        numbers_regulators = []
        self.gene_parser()
        for id_regulated, gene_regulated in enumerate(self.genes):
            number_regulators = 0
            for id_regulator, gene_regulator in enumerate(self.genes):
                if gene_regulated[self.regulatory_transcription_factor_idx] == gene_regulator[
                    self.transcription_factor_idx]:
                    connections.append((id_regulator, id_regulated))
                    number_regulators += 1
            numbers_regulators.append(number_regulators)
        return connections, numbers_regulators

    def regulate(self):
        # 0 means no voxel
        self.phenotype = np.zeros((self.cube_face_size, self.cube_face_size),
                                  dtype=object)  # x, y
        self.maternal_injection()
        self.growth()

    #####################################################################
    # Summary of the TF concentrations through all the cells 
    def summarize_tf_concentrations(self):
        summary = {}
        
        for cell in self.cells:
            for tf, concentrations in cell.transcription_factors.items():
                if tf not in summary:
                    summary[tf] = 0.0
                summary[tf] += sum(concentrations)

        return summary
    #####################################################################

    # develop embryo from single cell
    def growth(self):

        maximum_reached = False
        for t in range(0, self.dev_steps):

            # develops cells in order of age
            for idxc in range(0, len(self.cells)):

                cell = self.cells[idxc]
                self.increase(cell)
                ###############################################################
                #######################  DIFFUSION  ###########################
                if self.use_diffusion:
                    for tf in list(cell.transcription_factors.keys()):
                        self.intra_diffusion(tf, cell)
                        self.inter_diffusion(tf, cell)
                ###############################################################

                # try to grow new cell
                self.place_voxel(cell)

                # max voxels
                if self.quantity_voxels == self.max_voxels -1:
                    maximum_reached = True
                    break

                # do decay only after possible growth,
                # so that small increases have more chance to have an effect before decaying too much
                # this means that first injection/parsing decays a bit before expression,
                # but that is neglectable in comparison to injection size
                for tf in list(cell.transcription_factors.keys()):
                    self.decay(tf, cell)
                
            self.tf_history.append(self.summarize_tf_concentrations())

            if maximum_reached:
                break

    # increase of originally expressed genes (meaning that gene products resulting from diffusion/split do not increase)
    def increase(self, cell):

        # for all genes in the dna
        # TODO: easier to loop original_genes instead
        for idg, gene in enumerate(self.genes):

            # if that gene was originally expressed (during dna parse at cell split)
            if idg in cell.original_genes:

                # increases a genes tf if there is enough of its regulatory tf
                if cell.transcription_factors.get(gene[self.regulatory_transcription_factor_idx]):

                    tf_in_all_sites = sum(cell.transcription_factors[gene[self.regulatory_transcription_factor_idx]])

                    regulatory_min_val = min(float(gene[self.regulatory_v1_idx]),
                                             float(gene[self.regulatory_v2_idx]))
                    regulatory_max_val = max(float(gene[self.regulatory_v1_idx]),
                                             float(gene[self.regulatory_v2_idx]))

                    if tf_in_all_sites >= regulatory_min_val and tf_in_all_sites <= regulatory_max_val:
                        cell.transcription_factors[gene[self.transcription_factor_idx]][int(gene[self.diffusion_site_idx])] += \
                            float(gene[self.transcription_factor_amount_idx]) \
                            / float(self.increase_scaling)
                        

    ################################################################################################################################################################
    #################################################   DIFFUSION   ################################################################################################

    def inter_diffusion(self, tf, cell):

        for ds in range(0, self.diffusion_sites_qt):
            if cell.transcription_factors[tf][ds] < self.inter_diffusion_rate:
                continue

            neighbour_coord, neighbour_slot = self.find_child_slot(cell.xy_coordinates, ds)

            if not all(0 <= i < self.cube_face_size for i in neighbour_coord):
                continue

            neighbour = self.phenotype[tuple(neighbour_coord)]

            if neighbour == 0:
                continue

            cell.transcription_factors[tf][ds] -= self.inter_diffusion_rate

            if tf not in neighbour.transcription_factors:
                neighbour.transcription_factors[tf] = [0] * GRN.diffusion_sites_qt

            neighbour.transcription_factors[tf][neighbour_slot] += self.inter_diffusion_rate

    def intra_diffusion(self, tf, cell):

        for ds in range(0, self.diffusion_sites_qt):
            ds_left = ds - 1 if ds - 1 >= 0 else self.diffusion_sites_qt - 1
            ds_right = ds + 1 if ds + 1 <= self.diffusion_sites_qt - 1 else 0

            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_right] += self.intra_diffusion_rate

            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_left] += self.intra_diffusion_rate
    
    #################################################   DIFFUSION   ################################################################################################
    ################################################################################################################################################################


    def decay(self, tf, cell):
        # decay in all sites
        for ds in range(0, GRN.diffusion_sites_qt):
            cell.transcription_factors[tf][ds] = \
                max(0, cell.transcription_factors[tf][ds] - self.concentration_decay)

    def place_voxel(self, parent_cell):
        product_concentrations = []

        for idm in range(0, len(self.structural_products)-1): # excludes voxel_type offphase
            # sum concentration of all diffusion sites
            concentration = sum(parent_cell.transcription_factors[self.structural_tfs[idm]]) \
                if parent_cell.transcription_factors.get(self.structural_tfs[idm]) else 0
            product_concentrations.append(concentration)

        # chooses structural tf with the highest concentration
        idx_max = product_concentrations.index(max(product_concentrations))

        # if tf concentration above a threshold
        if product_concentrations[idx_max] > self.concentration_threshold:

            # grows in the free diffusion site with the highest concentration
            freeslots = np.array([c is None for c in parent_cell.children])
            if any(freeslots):
                true_indices = np.where(freeslots)[0]
                values_at_true_indices = np.array(parent_cell.transcription_factors[self.structural_tfs[idx_max]])[true_indices]
                max_value_index = np.argmax(values_at_true_indices)
                position_of_max_value = true_indices[max_value_index]
                slot = position_of_max_value

                potential_child_coord, child_slot = self.find_child_slot(parent_cell.xy_coordinates, slot)

                # if coordinates within cube bounderies and if position not occupied
                if all(0 <= i < self.cube_face_size for i in potential_child_coord):

                    if self.phenotype[tuple(potential_child_coord)] == 0:
                        key, voxel_type = list(self.structural_products.items())[idx_max]

                        # --- offphase alternation: starts with phase ---
                        if voxel_type == self.structural_products['phase_muscle']:
                            if self._phase_run >= self.offphase_alternation_k:
                                voxel_type = self.structural_products['offphase_muscle']
                                self._phase_run = 0
                            else:
                                self._phase_run += 1

                        self.quantity_voxels += 1
                        self.new_cell(voxel_type, parent_cell, slot, child_slot, potential_child_coord)

    def new_cell(self, voxel_type, parent_cell, parent_slot, child_slot, xy_coordinates):

        new_cell = Cell(voxel_type=voxel_type, parent_cell=parent_cell, xy_coordinates=xy_coordinates)
        self.phenotype[tuple(xy_coordinates)] = new_cell

        # share concentrations in diffusion site of parent with child
        for tf in parent_cell.transcription_factors:

            if parent_cell.transcription_factors[tf][parent_slot] > 0:
                half_concentration = parent_cell.transcription_factors[tf][parent_slot] / 2
                parent_cell.transcription_factors[tf][parent_slot] = half_concentration
                new_cell.transcription_factors[tf] = [0] * GRN.diffusion_sites_qt
                new_cell.transcription_factors[tf][child_slot] = half_concentration

        self.express_genes(new_cell)
        self.cells.append(new_cell)

    def find_child_slot(self, xy_coordinates_parent, parent_slot):

        x = 0
        y = 1

        if parent_slot == DS.LEFT:
            child_slot = DS.RIGHT
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[x] -= 1

        if parent_slot == DS.RIGHT:
            child_slot = DS.LEFT
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[x] += 1

        if parent_slot == DS.UP:
            child_slot = DS.DOWN
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[y] += 1

        if parent_slot == DS.DOWN:
            child_slot = DS.UP
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[y] -= 1

        return xy_coordinates_child, child_slot

    def maternal_injection(self):

        # injects maternal tf into zygot and starts development of the first cell
        # the tf injected is regulatory tf of the first gene in the genetic string
        # the amount injected is the minimum for the regulatory tf to regulate its regulated product
        first_gene_idx = 0
        tf_label_idx = 0
        min_value_idx = 1
        # TODO: do not inject nor grow if there are no genes (unlikely)
        mother_tf_label = self.genes[first_gene_idx][tf_label_idx]
        mother_tf_injection = float(self.genes[first_gene_idx][min_value_idx])

        # TODO: do we really need to force the first cell?
        middle_pos = [s // 2 for s in self.phenotype.shape]
        # chosen initial cell is offphase, to counteract phase bias of anternation
        first_cell = Cell(voxel_type=self.structural_products['offphase_muscle'],
                          parent_cell=None,
                          xy_coordinates=middle_pos)
        first_cell.xy_coordinates = middle_pos
        # distributes injection among diffusion sites
        first_cell.transcription_factors[mother_tf_label] = \
            [mother_tf_injection/GRN.diffusion_sites_qt] * GRN.diffusion_sites_qt

        self.express_genes(first_cell)
        self.cells.append(first_cell)
        self.phenotype[tuple(middle_pos)] = first_cell

    def express_genes(self, new_cell):

        for idg, gene in enumerate(self.genes):

            regulatory_min_val = min(float(gene[self.regulatory_v1_idx]),
                                     float(gene[self.regulatory_v2_idx]))
            regulatory_max_val = max(float(gene[self.regulatory_v1_idx]),
                                     float(gene[self.regulatory_v2_idx]))

            if new_cell.transcription_factors.get(gene[self.regulatory_transcription_factor_idx]):
                # expresses a gene if its regulatory tf is present and within a range
                tf_in_all_sites = sum(new_cell.transcription_factors[gene[self.regulatory_transcription_factor_idx]])
                if tf_in_all_sites >= regulatory_min_val and tf_in_all_sites <= regulatory_max_val:

                    # update or add
                    if new_cell.transcription_factors.get(gene[self.transcription_factor_idx]):
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] \
                            [int(gene[self.diffusion_site_idx])] += float(gene[self.transcription_factor_amount_idx])
                    else:

                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] = [0] * GRN.diffusion_sites_qt
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] \
                        [int(gene[self.diffusion_site_idx])] = float(gene[self.transcription_factor_amount_idx])

                    new_cell.original_genes.append(idg)


class Cell:

    def __init__(self, voxel_type, parent_cell, xy_coordinates):
        self.voxel_type = voxel_type
        self.transcription_factors = {}
        self.original_genes = []
        self.xy_coordinates = xy_coordinates
        self.parent_cell = parent_cell
        self.children = [None] * GRN.diffusion_sites_qt


class DS:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

# voxel perspective (2D)
# Np = [x, y]
# x: left/right
# y: up/down


###### operators ######

# init
def initialization(rng, ini_genome_size):

    genome_ini_size = ini_genome_size
    genome_size = genome_ini_size + 1
    genotype = [round(rng.uniform(0, 1), 2) for _ in range(genome_size)]
    return genotype


# unequal crossover (proportional)
def unequal_crossover_prop(
        rng,
        promoter_threshold,  # must match the param inside the GRN class
        max_geno_size,
        parent1,
        parent2,
):
    parent1 = parent1.genome
    parent2 = parent2.genome

    types_nucleotides = 6
    # the first nucleotide is the concentration
    new_genotype = [(parent1[0] + parent2[0]) / 2]
    p1 = parent1[1:]
    p2 = parent2[1:]

    # --- helper: find promoter indices in a parent genome (excluding concentration) ---
    def get_promoters(parent):
        promotor_sites = []
        nucleotide_idx = 0
        while nucleotide_idx < len(parent):
            if parent[nucleotide_idx] < promoter_threshold:
                # enough room after promoter to form a full gene
                if (len(parent) - 1 - nucleotide_idx) >= types_nucleotides:
                    promotor_sites.append(nucleotide_idx)
                    nucleotide_idx += types_nucleotides  # skip the gene we just found
            nucleotide_idx += 1
        return promotor_sites

    # ---------- FIRST PARENT: choose side randomly (head or tail) ----------
    promoters_p1 = get_promoters(p1)
    if promoters_p1:
        cut_p1 = rng.sample(promoters_p1, 1)[0]

        # randomly choose whether we take head (0..cut+gene) or tail (cut..end)
        take_head_p1 = rng.random() < 0.5

        if take_head_p1:
            # include the promoter and its full gene block (+ types_nucleotides), plus the nucleotide after gene (+1)
            subset_p1 = p1[0:cut_p1 + types_nucleotides + 1]
        else:
            # take from promoter cut to the end (tail), starting at the promoter index
            subset_p1 = p1[cut_p1:]
    else:
        # no promoters found; take nothing from first parent
        subset_p1 = []

    new_genotype += subset_p1

    #  compute the proportion actually taken from first parent
    # (relative to its whole genome, excluding concentration)
    #     - If no nucleotides, proportion is 0.0
    prop_from_p1 = (len(subset_p1) / len(p1)) if len(p1) > 0 else 0.0

    # ---------- SECOND PARENT: target complementary proportion on a chosen side ----------
    promoters_p2 = get_promoters(p2)

    # complementary proportion we want from parent 2
    target_prop_p2 = 1.0 - prop_from_p1
    target_len_p2 = int(round(target_prop_p2 * len(p2))) if len(p2) > 0 else 0

    # randomly decide if we aim for head (first) or tail (second) part of parent 2
    take_head_p2 = rng.random() < 0.5

    if promoters_p2 and len(p2) > 0:
        # pick a promoter cut that best matches the target length on the chosen side
        best_cut = None
        best_diff = None

        for c in promoters_p2:
            if take_head_p2:
                # length if we take head up to full-gene after promoter c
                seg_len = min(c + types_nucleotides + 1, len(p2))
            else:
                # length if we take tail from promoter c to the end
                seg_len = len(p2) - c

            diff = abs(seg_len - target_len_p2)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_cut = c

        cut_p2 = best_cut if best_cut is not None else promoters_p2[0]

        # apply the chosen side with the selected promoter cutpoint
        if take_head_p2:
            subset_p2 = p2[0: min(cut_p2 + types_nucleotides + 1, len(p2))]
        else:
            subset_p2 = p2[cut_p2:]
    else:
        # if no promoters (or empty), nothing from second parent
        subset_p2 = []

    new_genotype += subset_p2

    return new_genotype


# unequal crossover
def unequal_crossover(
        rng,
        promoter_threshold,  # make sure it matches th param inside the GRN class
        max_geno_size,
        parent1,
        parent2,
):
    parent1 = parent1.genome
    parent2 = parent2.genome

    types_nucleotides = 6
    # the first nucleotides are params
    params_nucleotides = 1
    new_genotype = [(parent1[0] + parent2[0]) / 2]
    p1 = parent1[params_nucleotides:]
    p2 = parent2[params_nucleotides:]

    for parent in [p1, p2]:
        nucleotide_idx = 0
        promotor_sites = []
        while nucleotide_idx < len(parent):
            if parent[nucleotide_idx] < promoter_threshold:
                # if there are nucleotides enough to compose a gene
                if (len(parent) - 1 - nucleotide_idx) >= types_nucleotides:
                    promotor_sites.append(nucleotide_idx)
                    nucleotide_idx += types_nucleotides
            nucleotide_idx += 1

        # TODO: allow uniform random choice of keeping material after cut point instead of up to it
        cutpoint = rng.sample(promotor_sites, 1)[0]
        subset = parent[0:cutpoint + types_nucleotides + 1]
        new_genotype += subset

    if len(new_genotype) > max_geno_size:
        new_genotype = new_genotype[0:max_geno_size]

    return new_genotype


# mutation for unequal crossover
def mutation_type1(rng, genome):

    position = rng.sample(range(0, len(genome)), 1)[0]
    type = rng.sample(['perturbation', 'deletion', 'addition', 'swap'], 1)[0]

    if type == 'perturbation':
        newv = round(genome[position] + rng.normalvariate(0, 0.1), 2)
        if newv > 1:
            genome[position] = 1
        elif newv < 0:
            genome[position] = 0
        else:
            genome[position] = newv

    if type == 'deletion':
        genome.pop(position)

    if type == 'addition':
        genome.insert(position, round(rng.uniform(0, 1), 2))

    if type == 'swap':
        position2 = rng.sample(range(0, len(genome)), 1)[0]
        while position == position2:
            position2 = rng.sample(range(0, len(genome)), 1)[0]

        position_v = genome[position]
        position2_v = genome[position2]
        genome[position] = position2_v
        genome[position2] = position_v

    return genome


# TEST
if __name__ == "__main__":
    import random

    rng = random.Random(3)
    genome = initialization(rng, ini_genome_size=80)

    cells = GRN(
        max_voxels=36,
        cube_face_size=6,
        genotype=genome,
        voxel_types="withbone",
        env_conditions="",
        plastic=0,
    ).develop()

    phenotype = np.zeros(cells.shape, dtype=int)
    for idx, value in np.ndenumerate(cells):
        phenotype[idx] = value.voxel_type if value != 0 else 0

    print("RANDOM GENOME LENGTH (GRN vector):", len(genome))
    print("BODY SHAPE/MATERIALS:")
    print(phenotype)
