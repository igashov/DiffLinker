import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig

# Set up features to use in FeatureMap
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable',
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    return fm_score


def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                                                     allowReordering=False)
    SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    return SC_RDKit_score