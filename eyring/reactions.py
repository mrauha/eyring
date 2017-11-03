import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import cclib as cc

from scipy.constants import physical_constants, Avogadro, calorie, kilo
hartree, _, _ = physical_constants["Hartree energy"]
kcalpermol = hartree * Avogadro / (calorie * kilo)


def _free_energy_comps(output):
    """
    Auxiliary function for obtaining free energy (G) and its components
    enthalpy (H) and entropy (S).

    Parameters
    ----------
    output : str
        A quantum chemistry output file name.

    Returns
    -------
    freeenergy, enthalpy, entropy : float
        A tuple of (absolute) free energy (G), enthalpy (H) and entropy (S) in
        hartree/particle, such that G = H - T * S.
    """

    d = cc.parser.ccopen(output)
    m = d.parse()

    return m.freeenergy, m.enthalpy, m.entropy


def _parse_step(step):
    """
    Auxiliary function for recursively parsing steps according to the
    specification of `mechanism`.

    Parameters
    ----------
    step : str, list or tuple
        A properly defined step for `mechanism`. See the description of the
        parameter `step` in `mechanism` for more details.

    Returns
    -------
    freeenergy, enthalpy, entropy : float
        A tuple of (absolute) free energy (G), enthalpy (H) and entropy (S) in
        hartree/particle, such that G = H - T * S.
    name : str
        A convenient name for the step. It is taken as the file name (for when
        isomers are given) or a concatenation of file names such as
        "filename1+filename2" (for multimolecular reactions mechanisms).
    """

    freeenergy, enthalpy, entropy = 0., 0., 0.
    name = ""

    if isinstance(step, str):
        freeenergy, enthalpy, entropy = _free_energy_comps(step)
        name = step
    elif isinstance(step, list):
        for isomer in step:
            _freeenergy, _enthalpy, _entropy, _name = _parse_step(isomer)

            if freeenergy > _freeenergy:
                freeenergy = _freeenergy
                enthalpy, entropy = _enthalpy, _entropy
                name = _name
    elif isinstance(step, tuple):
        for molecule in step:
            _freeenergy, _enthalpy, _entropy, _name = _parse_step(molecule)

            freeenergy += _freeenergy
            enthalpy += _enthalpy
            entropy += _entropy
            name = "{:s}+{:s}".format(name, _name)

        name = name[1:]
    else:
        raise TypeError("step type should be str, list or tuple")

    return freeenergy, enthalpy, entropy, name


def mechanism(steps, G=None):
    """
    Describe a reaction mechanism as a multiedged digraph representation. It
    reads from a set of quantum chemistry calculations as given as output file
    names in `steps`.

    Parameters
    ----------
    steps : (nested) list of either strings or lists of strings or tuples of
    either strings or list of strings
        A (nested) list whose each element represents a single step in a
        reaction mechanism. Each element of the list may be:
            i. a string representing a quantum chemistry calculation output;
            ii. a list representing different isomers of the same structure
            (the one with the smallest free energy will be used);
            iii. a tuple representing a multimolecular step (free energy will
            be taken as the sum of all structures).
        These descriptions can be nested (see examples below).
    G : networkx.DiGraph, optional
        If given, the reaction mechanism description will be written in `G`
        and. In this case, an updated version of `G` will be returned.

    Returns
    -------
    networkx.DiGraph
        A multiedged digraph that describes the mechanism.

    Examples
    --------

    >> mechanism(["a+b.out", "c.out"])
    >> mechanism([("a.out", "b.out"), "c.out"])
    >> mechanism([["a+b1.out", "a+b2.out"], "c.out"])
    >> mechanism([(["a1.out", "a2.out"], "b.out"), "c.out"])
    """

    # assert steps is a list

    if G is None:
        G = nx.DiGraph()

    last_name = None
    for step in steps:
        freeenergy, enthalpy, entropy, name = _parse_step(step)

        G.add_node(name, freeenergy=freeenergy, enthalpy=enthalpy,
                   entropy=entropy)

        if last_name is not None:
            G.add_edge(last_name, name)

        last_name = name

    return G


def diagram(G, source, target, names=None, relative_energies=None,
            energy_conversion=kcalpermol,
            energy_units=r"kcal$\cdot$mol$^{-1}$", diagram_width=None,
            step_width=1., transition_width=None, step_pattern="k-",
            transition_pattern="k--", color=None):
    """
    Draw a reaction diagram from `source` to `target` as a diagram using
    `matplotlib`.

    Parameters
    ----------
    G : networkx.DiGraph
        A multiedged digraph as returned from `mechanism`. The representation
        of reaction steps in the reaction path will be taken from nodes of `G`.
        Specifically, the attributes `freeenergy` and `name` will
        correspond to energy levels and their annotations.
    source : str
        Node name representing the reactant step.
    target : str
        Node name representing the product step.
    names : dict, optional
        A dict from log file names to names to show in the diagram.
    relative_energies : None, bool or str, optional
        If `True`, only relative energies will be used. If a string is given,
        it will represent a the name of a node whose energy will be subtracted
        from all other energies. If `None`, which is the default option, the
        string for a node name will be taken from `source`.
    energy_conversion : float, optional
        Multiplicative energy conversion to be applied. Default conversion is
        from hartree to kcal/mol.
    energy_units : str, optional
        Units of energy. It is used to properly set the y-axis and defaults to
        "kcal/mol". This parameter is related to energy_conversion above.
    diagram_width, step_width, transition_width : float, optional
        These parameters control the dimensions of steps, transitions and the
        final diagram. Both `step_width` and `transition_width` are equal to
        `1.` by default. By specifying `diagram_width`, `step_width` will be
        modified to honour the desired diagram dimension, while keeping
        `step_width` fixed.
    step_pattern, transition_pattern : str, optional
        Line pattern of steps and transitions as understandable by
        `matplotlib`.
    color : str, optional
        Single character representing colour as understandable by `matplotlib`.
        If set, it overwrites the first character of both `step_pattern` and
        `transition_pattern`.
    """

    freeenergies_dict = nx.get_node_attributes(G, "freeenergy")
    max_freeenergy = -np.inf
    min_freeenergy = np.inf

    for path in nx.all_simple_paths(G, source, target):
        path_freeenergies = []
        for step in path:
            path_freeenergies.append(freeenergies_dict[step])

        path_freeenergies = np.array(path_freeenergies)

        if relative_energies is None:
            relative_energies = source

        if isinstance(relative_energies, str):
            path_freeenergies = path_freeenergies \
                              - freeenergies_dict[relative_energies]
        elif relative_energies is True:
            path_freeenergies = path_freeenergies - np.min(path_freeenergies)

        path_freeenergies *= energy_conversion

        _min_freeenergy = np.min(path_freeenergies)
        _max_freeenergy = np.max(path_freeenergies)
        if min_freeenergy > _min_freeenergy:
            min_freeenergy = _min_freeenergy
        if max_freeenergy < _max_freeenergy:
            max_freeenergy = _max_freeenergy

        if color is not None:
            step_pattern = "{:s}{:s}".format(color, step_pattern[1:])
            transition_pattern = "{:s}{:s}".format(color,
                                                   transition_pattern[1:])

        nsteps = len(path_freeenergies)

        step_y_ranges = zip(path_freeenergies, path_freeenergies)
        transition_y_ranges = zip(path_freeenergies[:-1],
                                  path_freeenergies[1:])

        if transition_width is None:
            transition_width = step_width

        if diagram_width is not None:
            transition_width = (diagram_width - nsteps * step_width) \
                             / (nsteps - 1)

        step_x = (step_width + transition_width) * np.arange(nsteps)
        transition_x = step_x + step_width

        step_x_ranges = zip(step_x, step_x + step_width)
        transition_x_ranges = zip(transition_x,
                                  transition_x + transition_width)

        above_step = np.array([-.43e-2,  .20])
        below_step = np.array([-.63e-3, -.60])

        above_step *= step_width * mpl.rcParams["font.size"]
        below_step *= step_width * mpl.rcParams["font.size"]

        for i, (xx, yy) in enumerate(zip(step_x_ranges, step_y_ranges)):
            plt.plot(xx, yy, step_pattern)
            pos = np.array([np.average(xx), np.average(yy)])

            if names is not None and path[i] in names:
                note = names[path[i]]
                adjust = np.array([len(note), 1.])
                plt.annotate(note, pos + below_step * adjust)
            else:
                note = path[i]
                adjust = np.array([len(note), 1.])
                plt.annotate(note, pos + below_step * adjust)

            note = "{:.2f}".format(path_freeenergies[i])
            adjust = np.array([len(note), 1.])
            plt.annotate(note, pos + above_step * adjust)

        for xx, yy in zip(transition_x_ranges, transition_y_ranges):
            plt.plot(xx, yy, transition_pattern)

    axes = plt.gca()
    _min_freeenergy, _max_freeenergy = axes.get_ylim()
    if min_freeenergy > _min_freeenergy:
        min_freeenergy = _min_freeenergy
    if max_freeenergy < _max_freeenergy:
        max_freeenergy = _max_freeenergy

    scale = max_freeenergy - min_freeenergy
    pscale = .9e-2
    pscale *= step_width * mpl.rcParams["font.size"]

    plt.ylim(min_freeenergy - pscale * scale, max_freeenergy + pscale * scale)
    plt.ylabel(r"$\Delta G$ / {:s}".format(energy_units))
    plt.xlabel(r"Reaction coordinate")
    plt.xticks([])
