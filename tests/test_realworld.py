"""Test stores found in the real world."""

import pytest

URLS = {
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0010/76-45.ome.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0090/190129.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpD_chicken_embryo_MIP.ome.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0157/Asterella gracilis SWE/IMG_1033-1112 Asterella gracilis (Mannia gracilis) stature.ome.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0051/180712_H2B_22ss_Courtney1_20180712-163837_p00_c00_preview.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0026/3.66.9-6.141020_15-41-29.00.ome.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0083/9822152.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0033A/BR00109990_C2.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001247.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457537.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457227.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0044A/4007801.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0052A/5514375.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0047A/4496763.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0076A/10501752.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0054A/5025551.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0048A/9846151.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0079A/idr0079_images.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0072B/9512.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0056B/7361.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0128E/9701.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0013A/3451.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0001A/2551.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0040A/3491626.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0051A/4007817.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0052A/5514375.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0079A/9836998.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0094A/7751.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0095B/11511419.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0109A/12922361.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/yx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/zyx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/cyx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/tyx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/tcyx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/czyx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/tczyx.ome.zarr",
    "https://s3.embl.de/i2k-2020/ngff-example-data/v0.4/multi-image.ome.zarr",
    "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr",
    "https://s3.embl.de/eosc-future/EUOS/testdata.zarr",
    "https://s3.embl.de/ome-zarr-course/data/commons/xyz_8bit_calibrated__fib_sem_crop.ome.zarr",
    "s3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_mus-heart-1/jrc_mus-heart-1.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_mus-hippocampus-1/jrc_mus-hippocampus-1.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_cos7-1b/jrc_cos7-1b.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_jurkat-1/jrc_jurkat-1.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_mus-sc-zp105a/jrc_mus-sc-zp105a.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_sum159-1/jrc_sum159-1.zarr/recon-1/em/",
    "s3://janelia-cosem-datasets/jrc_mus-skin-1/jrc_mus-skin-1.zarr/recon-1/em/",
    "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BSST410/IM2/IM2.zarr",
    "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD463/00cddbf9-1282-4a49-94d9-481b7a43cc0c/00cddbf9-1282-4a49-94d9-481b7a43cc0c.zarr/",
    "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD1021/06bc50fb-03ae-4dc5-8a12-89d4f2fcbade/91e29e80-0467-428f-8d96-16cbee80b2fe.ome.zarr/0",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/aneurism.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/backpack.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/beechnut.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/bonsai.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/bunny.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/chameleon.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/engine.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/foot.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/kingsnake.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/lobster.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/marmoset_neurons.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/miranda.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/pig_heart.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/skull.ome.zarr",
    "https://ome-zarr-scivis.s3.us-east-1.amazonaws.com/v0.5/96x2/stag_beetle.ome.zarr",
}


@pytest.mark.parametrize("url", URLS)
def test_realworld_store(url):
    """Test opening real-world stores."""
    import zarr

    # Configure storage options for S3 anonymous access
    storage_options = {}
    if url.startswith("s3://"):
        storage_options = {"anon": True}

    group = zarr.open_group(url, storage_options=storage_options)
    print()
    print(group.metadata.zarr_format)
