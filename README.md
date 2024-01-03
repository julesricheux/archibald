# [archibald 1.0.0-beta](https://github.com/julesricheux/archibald/) :sailboat:✏️

archibald is a performance prediction package for sailboats and ships in the preliminary design stage. It is written in Python 3 and aimed in particular at naval architecture students.

It is still under continuous development, but modules are ready to be used independently.
In particular, **Rhino and AutoCAD models and drawings can be directly imported for computation**, giving an easy workflow to quickly assess the impact of design choices on performance, and play with it.

## Examples

Different examples of basic use can be found in /examples/:
- Hydrostatics and stability curve computation
- Hull resistance computation
- Appendage hydrodynamics computation
- Sails aerodynamics computation

## Dependencies

### For general use:
[NumPy](https://numpy.org/)
[SciPy](https://scipy.org/)
[Trimesh](https://trimesh.org/)
[ezdxf](https://ezdxf.mozman.at/)
[Shapely](https://pypi.org/project/shapely/)
[csv](https://docs.python.org/fr/3/library/csv.html)
[tqdm](https://github.com/tqdm/tqdm)
[Matplotlib](https://matplotlib.org/)

### For experimental features:
[AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)
[OpenPlaning](https://github.com/elcf/python-openplaning)

## Main features

- Holtrop and Delft hull resistance method
- Built-in mesh-based hydrostatics
- Perfect fluid-boundary layer coupling

## Experimental features

- Sails computation with vertical wind gradient
- Planing hull resistance computation
- Post-stall aerodynamics model

## License
[MIT License, terms here](LICENSE.txt)

## References
Scientific papers used to write this code can be found in /refs/
