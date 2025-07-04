def test_module_imports():
    import importlib

    modules = [
        "nids.pipelines.train",
        "nids.pipelines.predict",
        "nids.models.random_forest",
        "nids.utils.repair_cic_ids_csv",
    ]
    for mod in modules:
        assert importlib.import_module(mod) is not None 