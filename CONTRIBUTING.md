# Contributing to MasharikiWeather

Welcome to the MasharikiWeather development team! As an internal DSAIL tool, we encourage active experimentation and stress-testing. However, to keep our data engine robust and scientifically accurate, we follow standard software engineering practices.

## The Goal: Stress-Testing & Experimentation
We want you to break things, test new interpolations, and push the limits of spatiotemporal ML. If you need a new feature to support your specific experiment (e.g., a new data source, a custom PyTorch DataLoader, or a different spatial masking technique), please add it! 

Just follow the workflow below so we can integrate your brilliant ideas without breaking the core engine.

## Branching Strategy
Never commit directly to the `main` branch. Always create a new branch for your work. Use the following naming conventions:
* `feature/your-feature-name` (for new core capabilities - ideally to help in your experiments)
* `bugfix/issue-description` (for fixing broken code)

On your working environment, you can create and switch to a new branch all in one step using the following command:
```bash
git checkout -b feature/add-tamsat-support # create and switch to a new branch all in one step
```
Or for a bug fix:
```bash
git checkout -b bugfix/fix-imputation-method
```

## Development Workflow
1. **Create a Branch**: Use the naming convention above. Ideally, this is when you write code for your feature or bug fix on your working environment.
2. **Open an issue**: Before, writing code to github, open an issue on Github describing what you want to build or what bug you found. Use the issue templates provided.
3. **Make Changes**: Add your new feature or fix on your specific branch.
4. **Keep it modular**: Avoid modifying core files unless necessary. If that is the case and you were to add a new extraction method or computation method or a new feature within a method, try to integrate into the existing function using (like `get_gridded_for_stations`) explicit arguments rather than duplicating the entire class or function. Create new files or classes to extend functionality.
5. **Test**: Run `pytest` to ensure you haven't broken existing functionality. If you add a new feature, add a new test for it. **Testing of features is currently not available but it is a core functionality and will be added very soon**
6. **Commit**: Write a clear commit message explaining what you did.

## Pull Requests
1. **Push**: Push your branch to the remote repository.
2. **Pull Request (PR)** : Open a Pull Request against `main` for review and merging.
3. In the PR description, explicitly state:
    * What the code does
    * Which issue it resolves
    * Any breaking changes
    * Any tests you added or how you tested it (e.g "Extracted 2023 TAHMO data and verified shapes match").
4. **Review**: A maintainer (at least one member of the DSAIL weather team) will review your PR (check for scientific correctness, code readability and quality, and potentially data leakage) and request changes if needed.
5. **Merge**: Once the PR is approved, a maintainer will merge it into the `main` branch.

## Coding standards
1. **Type Hinting**: Use type hints for all functions and methods e.g `stations: List[str]`, `parameters: Optional[Union[str, Sequence[str]]]`, `imputation_method: Literal["locf", "zero", None]`. We recommend the usage of `Union` for multiple types and `Optional` for optional arguments.
2. **Docstrings**: Briefly document what your new methods do, especially the expected mathematical shapes of the inputs and outputs.
3. **Testing**: Test your code to ensure you haven't broken existing functionality. If you add a new feature, add a new test for it. **Testing of features is currently not available but it is a core functionality and will be added very soon**
4. **No hardcoded paths**: Avoid hardcoding paths to files. Use `os.path.join` or `pathlib.Path` to create paths. Please use relative paths or pass paths as arguments to the functions. Do not hardcode paths specific to your local machine or programming environment (e.g. `/home/teo/...`)

