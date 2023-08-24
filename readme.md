# OSSO dataset loader

This code enable the loading, processing and evaluation of the OSSO dataset released as a UKBiobank dataset return (release planned for november 2023).
We release, for 2400 subjects, pairs of skeleton and body meshes as well as the corresponding STAR parameters. Those meshes were generated from the subjects' DXA scans, see the [project page](https://osso.is.tue.mpg.de/) for details.

This code lets you:
- Visualize the mesh data (Requires our [UKBiobank dataset return](https://biobank.ndph.ox.ac.uk/ukb/docs.cgi?id=1))
- Compare our meshes silhouette with the corresponding DXA silhouettes. (Requires access to UK Biobank [link](https://www.ukbiobank.ac.uk/enable-your-research))
- We also release the code to extract the body and skeleton silhouette from a DXA scan. 

# Installation

```
python3.8 -m venv ukb_env 
source ukb_env/bin/activate
pip install git+https://github.com/MPI-IS/mesh.git 
pip install -r requirements.txt
pip install -e . 
```    

Finally, in config.py, set the paths to the ukbiobank dataset and the return dataset.


# Demo

## Data loading

Load a subject's dxa from the ukbiobank dataset: 
```python demos/load_dxa.py```

Load the subject's 3D meshes and STAR parameters from our dataset return:
```python demos/load_return.py```

## Data processing

Preprocess a DXA to extract the body and skeleton silhouette: ```python demos/dxa_process.py```

## Data evaluation

Visualize and evaluate the overlap between the meshes and the DXA scan: ```python demos/fit_eval.py```


# Citation

If you find this dataset & software useful in your research, please consider citing:

```
@inproceedings{Keller:CVPR:2022,
  title = {{OSSO}: Obtaining Skeletal Shape from Outside},
  author = {Keller, Marilyn and Zuffi, Silvia and Black, Michael J. and Pujades, Sergi},
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2022},
  month_numeric = {6}}
```

# License

This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE.txt](LICENSE.txt) file.


# Contact

For more questions, please contact osso@tue.mpg.de

For commercial licensing, please contact ps-licensing@tue.mpg.de