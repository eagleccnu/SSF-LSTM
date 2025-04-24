from Bio.PDB import PDBList, PDBParser
from scipy.spatial import distance
import numpy as np
import fileinput
import os


class Protein_bfactor:

    def __init__(self, pdbID, chainID, path='files/'):
        self.pdbid = pdbID
        self.chainid = chainID
        self.seq_name = pdbID + chainID
        self.path = path
        self.resname = []
        self.resID = []
        self.bfactor = []
        self.all_coordinates = []
        self.CA_coord = []

    def download_pdb(self):

        pdbl = PDBList()
        pdbl.retrieve_pdb_file(self.pdbid, file_format='pdb', pdir=self.path)

    def get_data_from_pdb(self):

        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        data = parser.get_structure(self.pdbid.lower(), self.path + 'pdb' + self.pdbid.lower() + '.ent')
        model = data.get_models()
        models = list(model)
        chains = list(models[0].get_chains())


        atoms_res = []

        for chain in chains:
            if chain.id == self.chainid:
                residue = list(chain.get_residues())

                for i in range(len(residue)):
                    if residue[i].get_id()[0] == ' ':
                        self.resname.append(residue[i].get_resname())
                        id = residue[i].get_id()
                        id = str(id[1]) + id[2]
                        self.resID.append(id.replace(' ', ''))
                        atoms = list(residue[i].get_atoms())
                        atoms_name = []
                        bfactor = []
                        coordinates = []

                        for j in range(len(atoms)):
                            atoms_name.append(atoms[j].get_name())
                            atoms_res.append(atoms[j].get_name())
                            bfactor.append(atoms[j].get_bfactor())
                            coordinates.append(np.round(atoms[j].get_coord().astype(float), 3).tolist())
                            self.all_coordinates.append(np.round(atoms[j].get_coord().astype(float), 3).tolist())

                        if 'CA' in atoms_name:
                            self.bfactor.append(bfactor[atoms_name.index('CA')])
                            self.CA_coord.append(coordinates[atoms_name.index('CA')])
                        else:
                            self.bfactor.append(-1)
                            self.CA_coord.append([])

        return atoms_res

    def split_str(self, site):
        b = []
        for i in site:
            if i[0].isalpha() and str.isdigit(i[1:]):
                b.append(i[0])
                b.append(i[1:])
            else:
                b.append(i)
        return b

    def abbr_reduce(self, long_abbr):
        dict_abbr_reduce = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
                            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
                            'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

        list_long_abbr = list(dict_abbr_reduce.keys())
        if long_abbr in list_long_abbr:
            return dict_abbr_reduce[long_abbr]
        else:
            return 'X'

    def one_hot(self, residue):
        res = {'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'GLU': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'GLN': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

        list_res = list(res.keys())
        if residue in list_res:
            return res[residue]
        else:
            return 'Error'

    def phys_chem_code(self, aa_char):
        dict_phys_chem_code = {'ALA': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
                               'GLY': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
                               'VAL': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
                               'LEU': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
                               'ILE': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
                               'PHE': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
                               'TYR': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
                               'TRP': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
                               'THR': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
                               'SER': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
                               'ARG': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
                               'LYS': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
                               'HIS': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
                               'ASP': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
                               'GLU': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
                               'ASN': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
                               'GLN': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
                               'MET': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
                               'PRO': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
                               'CYS': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]}

        return dict_phys_chem_code[aa_char]

    def aa_pc_phi_psi_area_reso_R(self):

        aa = [self.one_hot(i) for i in self.resname]
        pc = [self.phys_chem_code(i) for i in self.resname]

        phi_psi_area = []
        f = open(os.path.join(self.path, self.pdbid.lower() + '.txt'), 'r')
        for line in f.readlines():
            if len(line.split()):
                if line.split()[0] == 'ASG':
                    if line.split()[2] == self.chainid:
                        phi_psi_area.append(list(map(float, line.split()[7:10])))
        f.close()

        reso_R = []
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        protein = parser.get_structure(self.pdbid.lower(), self.path + 'pdb' + self.pdbid.lower() + '.ent')
        resolution = float(protein.header["resolution"])
        f = open(self.path + 'pdb' + self.pdbid.lower() + '.ent', 'r')
        flag = 0
        for line in f.readlines():
            if flag == 0:
                if line.split()[2:4] == ['R', 'VALUE']:
                    try:
                        r_value = float(line.split()[-1])
                        flag += 1
                    except:
                        continue
        reso_R.append(resolution)
        reso_R.append(r_value)

        reso_R = [reso_R] * len(aa)

        return np.array(aa), np.array(pc), np.array(phi_psi_area), np.array(reso_R)

    def get_packingdensity(self):

        pd = []
        for ca_coord in self.CA_coord:
            if ca_coord:
                c_dis = [round(distance.euclidean(ca_coord, coord), 4) for coord in self.all_coordinates]
                num = -1
                for dis in c_dis:
                    if dis <= 10:
                        num += 1
                pd.append(num)
            else:
                pd.append(-1)
        pd = [(i-min(pd))/(max(pd)-min(pd)) for i in pd]
        pd = np.array(pd)
        packingdensity = pd[:, np.newaxis]

        return packingdensity

    def create_fasta(self):

        aa_str = ''.join(self.resname)
        with open(os.path.join(self.path + self.seq_name + '.fasta'), 'w') as f:
            f.writelines(['>' + self.seq_name + '\n', aa_str])

    def downloadPSSM(self):

        cmd = '/home/amax/ncbi-blast/ncbi-blast-2.11.0+/bin/psiblast' \
              ' -query ' + os.path.join(self.path + self.seq_name + '.fasta') + \
              ' -db /home/amax/ncbi-blast/ncbi-blast-2.11.0+/db/uni/uni -evalue 0.001 -num_iterations 3' \
              ' -out_ascii_pssm ' + os.path.join(self.path + self.seq_name + '.pssm')
        os.system(cmd)


    def downloadHMM(self):

        os.chdir('/home/amax/hh-suite/build')
        cmd1 = 'hhblits -i ' + os.path.join(self.path + self.seq_name + '.fasta') + \
               ' -d /home/amax/Project/database/uniprot20_2013_03/uniprot20_2013_03 -n 2 -mact 0.01' \
               ' -oa3m ' + os.path.join(self.path + self.seq_name + '.a3m')
        cmd2 = 'hhmake -i ' + os.path.join(self.path + self.seq_name + '.a3m') + \
               ' -o ' + os.path.join(self.path + self.seq_name + '.hmm')
        os.system('export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH" && %s && %s' % (cmd1, cmd2))


    def find_line(self, file):
        f = open(os.path.join(self.path + self.seq_name + '.hmm'), 'r')
        for line, strin in enumerate(f):
            if strin.split() == ['#']:
                fline = line
        f.close()
        return fline

    def get_pssm_hmm(self):

        pssm = []
        finput = fileinput.input(os.path.join(self.path + self.seq_name + '.pssm'))
        for line, strin in enumerate(finput):
            if line > 2:
                str_ve = strin.split()[1:22]
                if len(str_ve) == 0:
                    break
                pssm.append(list(map(int, str_ve[1:])))
        finput.close()

        h = []
        hh = []
        hhm = []
        finput = fileinput.input(os.path.join(self.path + self.seq_name + '.hmm'))
        for line, strin in enumerate(finput):
            if line > self.find_line(self.seq_name) + 4:
                str_ve = strin.split()[0:22]
                if len(str_ve) > 1:
                    h.append(list(str_ve))
        for i in range(0, len(h), 2):
            hh.append(h[i] + h[i + 1])
        for item in hh:
            item = item[2:]
            for i in range(len(item)):
                if item[i] == '0':
                    item[i] = 1
                elif item[i] == '*':
                    item[i] = 0
                else:
                    item[i] = round(2 ** (int(item[i]) / (-1000)), 2)
            hhm.append(item)
        finput.close()

        return np.array(hhm), np.array(pssm)

    def preprocess_data(self, features, seg_len=15):

        new_features, new_labels = [], []
        blank_res = [0] * 83 + [1]
        blank_segment = [blank_res] * ((seg_len - 1) // 2)
        flag = np.zeros((len(features), 1))
        features = np.concatenate((features, flag), axis=1)
        features = blank_segment + features.tolist() + blank_segment

        for i in range(len(features) - (seg_len - 1)):
            if self.bfactor[i] != 9999:
                new_features.append(features[i:seg_len + i])
                new_labels.append(self.bfactor[i])

        return np.array(new_features), new_labels





