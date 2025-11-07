
###### Functions for dealing with chemical structures and so on. Written by CM


import sys
import pandas as pd

from helpers import helper_general

#import deepchem as dc
import datamol as dm

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors, rdPartialCharges, QED
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger  

from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator, Descriptors
# from mordred import Calculator, descriptors

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from tqdm import tqdm

def get_RDKit_fp_types():
    return ["Atom pair", "Morgan", "RDKitFingerprint", "TopologicalTorsion"]

def get_deepchem_fp_types():
    return  ["circular_fp", "maccs_keys_fp", "pubchem_fp"]

def compute_RDKit_fingerpints(fp_type, cmols, fp_len=1024, get_numpy_version=True):
    list_fps = []
    try:
        print("generating fps", fp_type)
        
        generator = None
        if fp_type == "Atom pair":
            generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_len)
        elif fp_type == "Morgan":
            generator = rdFingerprintGenerator.GetMorganGenerator(radius = 5, fpSize=fp_len)          
        elif fp_type == "RDKitFingerprint":
            generator = rdFingerprintGenerator.GetRDKitFPGenerator(useHs=True, minPath=1, maxPath=10, fpSize=fp_len)            
        elif fp_type == "TopologicalTorsion":
            generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_len)
        
        fp_gen_function = generator.GetFingerprint
        if get_numpy_version:
            fp_gen_function = generator.GetFingerprintAsNumPy
        
        for i in tqdm(range(len(cmols))):
            fp = []
            try:
                fp = fp_gen_function(cmols[i])
            except:
                fp = [0 for i in range(fp_len)]
                
            list_fps.append(fp)  
            
    except Exception as ex:
        print(sys._getframe().f_code.co_name, str(ex))
    
    return list_fps

def compute_RDKit_MolDescriptors( cmols ):
    
    print("generate molecular descriptors")
    desc = []
    
    try:
        nms = [ x[0] for x in Descriptors._descList ]
        print(nms)
        calc = MoleculeDescriptors.MolecularDescriptorCalculator( nms )
        
        for i in tqdm(range(len(cmols))):
            moldesc = calc.CalcDescriptors(cmols[i])
            # if not any(np.isnan(moldesc)):
            desc.append(list(moldesc)) 
    except Exception as ex:
        print(str(ex))
        
    return desc

def compute_mordred(cmols):

    print("generate mordred descriptors")
    desc = []
    
    try:
        
        calc = Calculator(descriptors, ignore_3D=True)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                
        df = calc.pandas(cmols).select_dtypes(include=numerics)
        
        for i in tqdm(range(len(df))):
            desc.append(list(df.iloc[i].to_list())) 
                        
    except Exception as ex:
        print(str(ex))
        
    return desc    

def compute_deepchem(fp_type, cmols):
    
    print("generate deepchem descriptors")
    desc = []
    
    try:
        
        generator = None
        if fp_type == "circular_fp":        
            generator = dc.feat.CircularFingerprint(size=1024)
        elif fp_type == "maccs_keys_fp":
            generator = dc.feat.MACCSKeysFingerprint()
        elif fp_type == "pubchem_fp":
            print("it is very slow. Do they send the molecules to a pubchem ws api?")
        #    generator = dc.feat.PubChemFingerprint()
          
        fps = generator(cmols)
        
        for i in tqdm(range(len(fps))):
            desc.append(list(fps[i])) 
                        
    except Exception as ex:
        print(str(ex))
        
    return desc 

def calculate_descriptor(type_descriptor, list_mols, is_smiles=False, get_rdkit_numpy=True):
    
    list_desc = []

    try:
        if is_smiles:
            remover = SaltRemover()
            list_mols_new = []
            for smi in list_mols:
                try:
                    mol = remover.StripMol(Chem.MolFromSmiles(smi))
                except:
                    mol = None
                
                list_mols_new.append(mol)    
                
            list_mols = list_mols_new        
                
        if type_descriptor == "rdkit_moldesc":
            list_desc = compute_RDKit_MolDescriptors(list_mols)
        elif type_descriptor == "rdkit_fp":
            fp_types = get_RDKit_fp_types()
            list_desc = compute_RDKit_fingerpints(fp_types[1], list_mols, 1024, get_rdkit_numpy) 
        elif type_descriptor == "mordred":
            list_desc = compute_mordred(list_mols)     
        elif type_descriptor == "deepchem":
            fp_types = get_deepchem_fp_types()
            list_desc = compute_deepchem(fp_types[1], list_mols) 
                            
    except Exception as ex:
        print(sys._getframe().f_code.co_name, str(ex))
    
    return list_desc

def count_hbd_hba_atoms(m):
    
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                        '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                        '$([nH0,o,s;+0])]')     
    
    HDonor = m.GetSubstructMatches(HDonorSmarts)
    HAcceptor = m.GetSubstructMatches(HAcceptorSmarts)
    return len(set(HDonor + HAcceptor))

def annotate_pains(df, df_pains):
    
    tl = []
    
    try:
        
        
        list_mols = get_molecules_from_list_smiles(df["smiles"])
        list_smarts_mols = get_molecules_from_list_smarts(df_pains["smarts"])
        list_smarts_info = df_pains["info"]
        
        # print(list_smarts_mols)

        tl_tmp = Parallel(n_jobs=-1)(delayed(get_substruct_matches)(i, list_mols[i], list_smarts_mols) for i in tqdm(range(len(list_mols)), desc="PAINS search"))    
        
        print("tl_tmp", tl_tmp)
                
        df["pains"] = ["" for i in range(len(df))]
        #df["pains_info"] = ['' for i in range(len(df))]
        for tl_mol in tl_tmp:
            if len(tl_mol) > 0:
                print(tl_mol)                
                tl.append(tl_mol)    
                df["pains"].iloc[tl_mol[0][0]] = "#".join([list_smarts_info[tl_mol[j][1]] for j in range(len(tl_mol))])
                #df["pains_info"].iloc[tl_mol[0][0]] = " "
                #tl.append(tl_mol)
        
    except Exception as ex:
        print(str(ex))
    
    return tl

def get_substruct_matches(mol_id, mol, list_query_mols):
    
    matches = []
    
    try:
        
        for i in range(len(list_query_mols)):
            query_mol = list_query_mols[i]
            if query_mol is not None and mol is not None:
                match = mol.GetSubstructMatches(query_mol)
            
                if len(match) > 0:
                    matches.append((mol_id, i, match))
        
    except Exception as ex:
        print(str(ex))
    
    return matches

@helper_general.timing
def annotate_magic_rings(df, df_magic_rings):
    
    tl = []
    
    try:
        
        list_mols = get_molecules_from_list_smiles(df["smiles"])
        list_magic_mols = get_molecules_from_list_smiles(df_magic_rings["smiles"])

        tl_tmp = Parallel(n_jobs=-1)(delayed(get_substruct_matches)(i, list_mols[i], list_magic_mols) for i in tqdm(range(len(list_mols)), desc="Magic rings search"))    
        
        print(len(tl_tmp))
        for tl_mol in tl_tmp:
            if len(tl_mol) > 0:
                #print(tl_mol)
                tl.append(tl_mol)
        
    except Exception as ex:
        print(str(ex))
        
    return tl

def get_molecule_from_smarts(smarts):
    
    try:
        return Chem.MolFromSmarts(smarts)
    except:
        return None

def get_molecules_from_list_smarts(list_smarts):
    
    tl = []
    
    try:
        
        tl = Parallel(n_jobs=-1)(delayed(get_molecule_from_smarts)(list_smarts[i]) for i in tqdm(range(len(list_smarts))))    
        
    except Exception as ex:
        print(str(ex))
        
    return tl

def get_molecule_from_smiles(smiles):
    
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def get_molecules_from_list_smiles(list_smiles):
    
    tl = []
    
    try:
        
        tl = Parallel(n_jobs=-1)(delayed(get_molecule_from_smiles)(list_smiles[i]) for i in tqdm(range(len(list_smiles))))    
        
    except Exception as ex:
        print(str(ex))
        
    return tl

def get_physchem_props_from_smiles_rdkit(smiles):
    
    mol_prop_dict = {} 
    
    try:

        if "unknown_" in smiles:
            return mol_prop_dict
        
        mol_prop_dict["SMILES"] = str(smiles)
        
        if str(smiles) == "nan" or len(smiles) == 0:
            return {}
        
        m = Chem.MolFromSmiles(smiles)
        
        if m is not None:
            hba = rdMolDescriptors.CalcNumHBA(m)
            hbd = rdMolDescriptors.CalcNumHBD(m)
            nrings = rdMolDescriptors.CalcNumRings(m)
            rtb = rdMolDescriptors.CalcNumRotatableBonds(m)
            psa = rdMolDescriptors.CalcTPSA(m)
            logp, mr = rdMolDescriptors.CalcCrippenDescriptors(m)
            mw = rdMolDescriptors._CalcMolWt(m)
            csp3 = rdMolDescriptors.CalcFractionCSP3(m)
            hac = m.GetNumHeavyAtoms()        
            qed = QED.qed(m)        
            n_unique_hba_hbd_atoms = count_hbd_hba_atoms(m)

            core = MurckoScaffold.GetScaffoldForMol(m)
            murko_smi = Chem.MolToSmiles(core)
            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            murko_generic_smi = Chem.MolToSmiles(fw)
            
            mol_prop_dict["#H-bond acceptor"] = hba
            mol_prop_dict["#H-bond donor"] = hbd
            mol_prop_dict["#Rings"] = nrings
            mol_prop_dict["#rotatable bonds"] = rtb
            mol_prop_dict["Polar surface area"] = psa
            mol_prop_dict["LogP"] = logp
            mol_prop_dict["Molecular refractivity"] = mr
            mol_prop_dict["Molecular weight"] = mw
            mol_prop_dict["csp3"] = csp3
            mol_prop_dict["#Heavy atoms"] = hac
            mol_prop_dict["QED"] = qed
            mol_prop_dict["#unique hba hbd atoms"] = n_unique_hba_hbd_atoms
            mol_prop_dict["scaffold_smiles"] = murko_smi
            mol_prop_dict["scaffold_generic_smiles"] = murko_generic_smi
        
    except Exception as ex:
        print("MOL PROP CALC FAILED", str(ex))
    
    return mol_prop_dict

def get_physchem_props_from_smiles_dm(smiles):
    
    mol_prop_dict = {} 
    
    try:

        RDLogger.DisableLog('rdApp.*')
        
        if "unknown_" in smiles:
            return mol_prop_dict
        
        if str(smiles) == "nan" or len(smiles) == 0:
            return {}        
        
        RDLogger.DisableLog('rdApp.*')
        
        mol = dm.to_mol(smiles)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol)
        mol = dm.standardize_mol(mol)        
                      
        mol_prop_dict = dm.descriptors.compute_many_descriptors(mol)
        mol_prop_dict["smiles"] = str(smiles)
        
        #print(prop_df)
                
        #mol_prop_dict = prop_df.to_dict()
        
        
        # print(mol_prop_dict)
        
        if mol is not None:
            
            mol_prop_dict["#Heavy atoms"] = mol_prop_dict["n_heavy_atoms"]

            mol_prop_dict["Molecular weight"] = mol_prop_dict["mw"]

            logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
            mol_prop_dict["LogP"] = logp
            mol_prop_dict["Molecular refractivity"] = mr

            hba = rdMolDescriptors.CalcNumHBA(mol)
            mol_prop_dict["#H-bond acceptor"] = hba

            psa = rdMolDescriptors.CalcTPSA(mol)
            mol_prop_dict["Polar surface area"] = psa

            hbd = rdMolDescriptors.CalcNumHBD(mol)
            mol_prop_dict["#H-bond donor"] = hbd

            rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            mol_prop_dict["#rotatable bonds"] = rtb

            core = MurckoScaffold.GetScaffoldForMol(mol)
            murko_smi = Chem.MolToSmiles(core)

            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            murko_generic_smi = Chem.MolToSmiles(fw)
            mol_prop_dict["scaffold_smiles"] = murko_smi
            mol_prop_dict["scaffold_generic_smiles"] = murko_generic_smi

        # if m is not None:          
        #     mol_prop_dict["#H-bond acceptor"] = hba
        #     mol_prop_dict["#H-bond donor"] = hbd
        #     mol_prop_dict["#Rings"] = nrings
        #     mol_prop_dict["#rotatable bonds"] = rtb
        #     mol_prop_dict["Polar surface area"] = psa
        #     mol_prop_dict["LogP"] = logp
        #     mol_prop_dict["Molecular refractivity"] = mr
        #     mol_prop_dict["Molecular weight"] = mw
        #     mol_prop_dict["csp3"] = csp3
        #     mol_prop_dict["#Heavy atoms"] = hac
        #     mol_prop_dict["QED"] = qed
        #     mol_prop_dict["#unique hba hbd atoms"] = n_unique_hba_hbd_atoms
            # mol_prop_dict["scaffold_smiles"] = murko_smi
        #     mol_prop_dict["scaffold_generic_smiles"] = murko_generic_smi
        
    except Exception as ex:
        print("MOL PROP CALC FAILED", str(ex))
    
    return mol_prop_dict

def get_list_physchem_props_from_smiles(list_smiles):
    
    tl = []
    
    try:        
        tl = Parallel(n_jobs=-1)(delayed(get_physchem_props_from_smiles_dm)(list_smiles[i]) for i in tqdm(range(len(list_smiles))))    
    except Exception as ex:
        print('An exception occurred', str(ex)) 
    
    return tl

def get_datamol_smiles(smiles):
    
    smiles_clean = ""
    
    try:
        RDLogger.DisableLog('rdApp.*')
        
        if "unknown_" in smiles:
            return smiles
        
        if str(smiles) == "nan" or len(smiles) == 0:
            return ""    
        
        mol = dm.to_mol(smiles)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol)
        mol = dm.standardize_mol(mol)      
        
        smiles_clean = dm.to_smiles(mol)
        
    except Exception as ex:
        print(str(ex))
    
    return smiles_clean
    
def get_number_lipinski_ro5(df):
    
    tl = []
    try:
    
        # No more than 5 hydrogen bond donors (the total number of nitrogen–hydrogen and oxygen–hydrogen bonds)
        # No more than 10 hydrogen bond acceptors (all nitrogen or oxygen atoms)
        # A molecular mass less than 500 daltons
        # A calculated octanol-water partition coefficient (Clog P) that does not exceed 5

        for i in tqdm(range(len(df))):
            c_count = 0
            if df["#H-bond donor"].iloc[i] <= 5:
                c_count += 1
                
            if df["#H-bond acceptor"].iloc[i] <= 10:
                c_count += 1
            
            if df["Molecular weight"].iloc[i] <= 500:
                c_count += 1
                
            if df["LogP"].iloc[i] <= 5:
                c_count += 1
            
            if c_count > 3:
                tl.append(i)
    
    except Exception as ex:
        print(str(ex))
    
    return tl

def get_number_druglikeness_ghose(df):
    
    tl = []
    try:
        
        # Partition coefficient log P in −0.4 to +5.6 range
        # Molar refractivity from 40 to 130
        # Molecular weight from 180 to 480
        # Number of atoms from 20 to 70 (includes H-bond donors [e.g. OHs and NHs] and H-bond acceptors [e.g. Ns and Os])
        # 10 or fewer rotatable bonds and
        # Polar surface area no greater than 140 Å2
                 
        for i in tqdm(range(len(df))):
            
            if df["LogP"].iloc[i] >= -0.4 and df["LogP"].iloc[i] <= 5.6 and\
               df["Molecular refractivity"].iloc[i] >= 40 and df["Molecular refractivity"].iloc[i] <= 130 and\
               df["Molecular weight"].iloc[i] >= 180 and df["Molecular weight"].iloc[i] <= 480 and\
               df["#Heavy atoms"].iloc[i] >= 20 and df["#Heavy atoms"].iloc[i] <= 70:
                tl.append(i)
    
    except Exception as ex:
        print(str(ex))
    
    return tl

def get_number_druglikeness_veber(df):
    
    tl = []
    try:
        # 10 or fewer rotatable bonds and
        # Polar surface area no greater than 140 Å2
                 
        for i in tqdm(range(len(df))):
            
            if df["#rotatable bonds"].iloc[i] <= 10 and\
               df["Polar surface area"].iloc[i] <= 140:
                tl.append(i)
    
    except Exception as ex:
        print(str(ex))
    
    return tl

def smi2svg(smi, x=210, y=210):
    
    svg = ""
    
    try:
        if "unknown_" not in smi:
            mol = Chem.MolFromSmiles(smi)
            rdDepictor.Compute2DCoords(mol)
            mc = Chem.Mol(mol.ToBinary())
            Chem.Kekulize(mc)
            drawer = Draw.MolDraw2DSVG(x, y)
            drawer.DrawMolecule(mc)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace('svg:','')    
    except Exception as ex:
        pass
        #print(str(ex))    
        
    return svg 

def generate_molecule_images(list_smiles):
    
    tl_images = []
    
    try:
        
        tl_images = Parallel(n_jobs=-1)(delayed(smi2svg)(list_smiles[i]) for i in tqdm(range(len(list_smiles))))
        
    except Exception as ex:
        print(str(ex))
    
    return tl_images

def smi_match2svg(smi, matches, x=210, y=210):
    svg = ""
    
    try:
        mol = Chem.MolFromSmiles(smi)
        rdDepictor.Compute2DCoords(mol)
        
        print(smi)
        print(matches)
        print(matches[0][2])

        drawer = Draw.MolsToGridImage([mol],highlightAtomLists=matches[0][2], useSVG=True)

        # drawer = Draw.MolDraw2DSVG(210,210)

        # drawer = SimilarityMaps.GetSimilarityMapFromWeights(
        #     mol, 
        #     [x for x,y in contribs], 
        #     #[x for x in contribs], 
        #     colorMap='coolwarm', 
        #     contourLines=10,
        #     draw2d=drawer
        # ) 
        #print("TEST", drawer)
        
        #drawer.FinishDrawing()
        #svg = drawer.GetDrawingText().replace('svg:','') 
        
        #svg = fig
        #mc = Chem.Mol(fig)
        # Chem.Kekulize(mc)
        
        # drawer = Draw.MolDraw2DSVG(200,200)
        
        # drawer.FinishDrawing()
        # svg = drawer.GetDrawingText().replace('svg:','')    
        #print(svg)
    except Exception as ex:
        print("ERROR", str(ex))    
        
    return drawer 

def smi2svg_logp(smi):
    svg = ""
    
    try:
        mol = Chem.MolFromSmiles(smi)
        rdDepictor.Compute2DCoords(mol)
        
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        # print(contribs)
        # contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        # print(contribs)
        # #contribs = rdMolDescriptors._CalcLabuteASAContribs(mol) 
        # contribs = rdPartialCharges.ComputeGasteigerCharges(mol)
        # print(contribs)
        
        #contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
        
        drawer = Draw.MolDraw2DSVG(210,210)

        drawer = SimilarityMaps.GetSimilarityMapFromWeights(
            mol, 
            [x for x,y in contribs], 
            #[x for x in contribs], 
            colorMap='coolwarm', 
            contourLines=10,
            draw2d=drawer
        ) 
        #print("TEST", drawer)
        
        #drawer.FinishDrawing()
        #svg = drawer.GetDrawingText().replace('svg:','') 
        
        #svg = fig
        #mc = Chem.Mol(fig)
        # Chem.Kekulize(mc)
        
        # drawer = Draw.MolDraw2DSVG(200,200)
        # drawer.DrawMolecule(fig)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:','')    
        #print(svg)
    except Exception as ex:
        print("ERROR", str(ex))    
        
    return svg 
    
def fp_similarity_search(smiles, list_smiles, add_moldepiction=False):
    
    similarities = []
    try:
        
        qmol = Chem.MolFromSmiles(smiles)
        mols = [Chem.MolFromSmiles(smi) for smi in list_smiles]
        qmol_fp = calculate_descriptor("rdkit_fp", [smiles], True, False)
        fps = calculate_descriptor("rdkit_fp", list_smiles, True, False)               
        
        for i in range(len(fps)):
            
            fp = fps[i]
            
            similarity = DataStructs.FingerprintSimilarity(qmol_fp[0], fp)                        
            
            
            if add_moldepiction:
                drawer = Draw.MolDraw2DSVG(210,210)
                
                fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(
                    qmol, 
                    mols[i], 
                    SimilarityMaps.GetMorganFingerprint,
                    draw2d=drawer
                )                
                
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText().replace('svg:','')  
                print(similarity)
                similarities.append((similarity, svg))
            else:
                similarities.append(similarity)
            
    except Exception as ex:
        print(str(ex))
        
    return similarities

def get_similarities_as_coords(df):
    
    df_xy = pd.DataFrame()
    
    try:

        fps = calculate_descriptor("rdkit_fp", df["smiles"], is_smiles=True, get_rdkit_numpy=True)
        print(len(fps), len(df))
        if len(fps) > 0:            
            pca = PCA(n_components=2)
            tl_tmp = pca.fit_transform(fps)        
                    
            df_xy["X"] = [x[0] for x in tl_tmp]
            df_xy["Y"] = [x[1] for x in tl_tmp]
            
            km = KMeans(
                n_clusters=5, init='random',
                n_init=10, max_iter=300, 
                tol=1e-04, random_state=0
            )
            
            df_xy["clusters"] = km.fit_predict(fps)
                        
    except Exception as ex:
        print(sys._getframe().f_code.co_name, str(ex))
        
    return df_xy
    
def calculate_rdkit_descriptors_worker(smi, descriptor_names):
   
    tmp_dict = {}
   
    try:
      
        RDLogger.DisableLog('rdApp.*')
      
        mol = dm.to_mol(smi)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol)
        mol = dm.standardize_mol(mol)

        smi_clean = dm.to_smiles(mol)
        
        tmp_dict["smiles_old"] = smi
        tmp_dict["smiles_new"] = smi_clean
        
        for desc in descriptor_names:
         
            try:
                tmp_dict[desc] = dm.descriptors.any_rdkit_descriptor(desc)(mol)
            except Exception as ex:
                print(str(ex))              
    except Exception as ex:
        pass
   
    return tmp_dict    

def get_dataframe_rdkit_desc_from_smiles(list_smiles):
    
    df = pd.DataFrame()
    
    try:        
        
        descriptor_names = [desc[0] for desc in Descriptors._descList if not desc[0].startswith('fr_')]        
        
        # additional_desc = ['Crippen', 'LabuteASA', 'TPSA', 'NumLipinskiHBD', 'NumLipinskiHBA', 'NumAmideBonds',
        #                    'NumAtoms', 'NumStereocenters', 'NumUnspecifiedStereocenters', 'MQNs']

        additional_desc = ['CalcLabuteASA', 'CalcTPSA', 'CalcNumLipinskiHBD', 'CalcNumLipinskiHBA', 'CalcNumAmideBonds',
                           'CalcNumAtoms', 'CalcNumAtomStereoCenters', 'CalcNumUnspecifiedAtomStereoCenters', 'CalcCrippenDescriptors'] #, 'MQNs_']

        for i in range(len(additional_desc)):
            descriptor_names.append(additional_desc[i])                    
        
        results = Parallel(n_jobs=-1)(delayed(calculate_rdkit_descriptors_worker)(list_smiles[i], descriptor_names) for i in tqdm(range(len(list_smiles))))    
        
        for res in results:
            res["CalcCrippenDescriptors_logp"] = res["CalcCrippenDescriptors"][0]
            res["CalcCrippenDescriptors_mr"] = res["CalcCrippenDescriptors"][1]
            res.pop("CalcCrippenDescriptors")
        
        df = pd.DataFrame(results)        
                
    except Exception as ex:
        print('An exception occurred', str(ex)) 
        
    return df