# Automated Camera Trapping Identification and Organization Network (ACTION)

## Setup

Action is written in Python and requires a number of dependencies and large machine learning models (~778M) to be installed and downloaded.

The easiest way to use it is with the [pixi](https://prefix.dev/docs/pixi/overview) package manager.  Pixi installs everything you need into a local `.pixi` folder (i.e., at the root of the project), without needing to modify your system.

1. Clone this repo using `git clone https://github.com/humphrem/action.git`
2. [Install pixi](https://prefix.dev/docs/pixi/overview#installation)
3. Start a terminal and navigate to the root of the Action project folder you just cloned, `cd action`
4. Enter the command `pixi run setup` to download, install, and setup everything you'll need

## Using Action

### Pixi Shell

Each time you want to use Action, you need to open a terminal and navigate to the Action folder, then start a shell with `pixi`:

```sh
pixi shell
```

This will make all of the dependencies installed with `pixi run setup` available.

You can exit the pixi shell by using:

```sh
exit
```
