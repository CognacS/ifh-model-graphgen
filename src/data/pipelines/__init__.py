import src.data.pipelines.download_molecules as dw_mol
import src.data.pipelines.download_graphs as dw_gra
import src.data.pipelines.preprocessing_molecules as pp_mol
import src.data.pipelines.preprocessing_graphs as pp_gra

REGISTERED_DATATYPES = {
    'qm9': 'molecular',
    'zinc250k': 'molecular',
    'community-20': 'graph',
    'planar': 'graph',
    'sbm': 'graph',
    'ego-small': 'graph',
    'ego': 'graph',
    'enzymes': 'graph',
    'grid': 'graph'
}

REGISTERED_DOWNLOAD_PIPELINES = {
    'qm9': dw_mol.qm9_download_pipeline,
    'zinc250k': dw_mol.ZincDownloadPipeline('250k'),
    'community-20': dw_gra.CDGSDatasetDownloadPipeline('community-20'),
    'planar': dw_gra.SpectreDatasetDownloadPipeline('planar'),
    'sbm': dw_gra.SpectreDatasetDownloadPipeline('sbm'),
    'ego-small': dw_gra.CDGSDatasetDownloadPipeline('ego-small'),
    'ego': dw_gra.CDGSDatasetDownloadPipeline('ego'),
    'enzymes': dw_gra.CDGSDatasetDownloadPipeline('enzymes'),
    'grid': dw_gra.GRANDatasetGenerator()
}

REGISTERED_PREPROCESS_PIPELINES = {
    'qm9': pp_mol.qm9_preprocessing_pipeline,
    'zinc250k': pp_mol.ZincPreprocessingPipeline('250k'),
    'community-20': pp_gra.PickledDatasetPreprocessingPipeline('community-20'),
    'planar': pp_gra.SpectreDatasetPreprocessingPipeline('planar'),
    'sbm': pp_gra.SpectreDatasetPreprocessingPipeline('sbm'),
    'ego-small': pp_gra.PickledDatasetPreprocessingPipeline('ego-small'),
    'ego': pp_gra.PickledDatasetPreprocessingPipeline('ego'),
    'enzymes': pp_gra.PickledDatasetPreprocessingPipeline('enzymes'),
    'grid': pp_gra.PickledDatasetPreprocessingPipeline('grid')
}