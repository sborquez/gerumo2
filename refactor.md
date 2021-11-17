Refactor
========

The objetive of this refactor are:
* Improve configuration files: By using YAML, we can add comments to explain the configuration.
* More flexibility with model definitions, current model and ensemble implementation doesn't allow custom training or prediction functions.
* Cleaning unused code.
* Include new tasks, e.g. classification.
* Adding Tensorboard support for easily model monitoring.

## Millestones

- [x] Cleaning and renaming
- [ ] Use Dataset and DataLoader approach
- [ ] Support Prodb5
- [ ] Improve models definitions.
- [ ] YAMLs replace JSON
- [ ] Replace model monitoring
