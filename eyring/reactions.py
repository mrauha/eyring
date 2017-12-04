#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import cclib as cc

from scipy.constants import physical_constants, Avogadro, Boltzmann, Planck, \
    gas_constant, calorie, kilo
hartree, _, _ = physical_constants["Hartree energy"]
joulespermol = hartree * Avogadro
kcalpermol = joulespermol / (calorie * kilo)


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
    nimagfreqs : int
        The number of imaginary frequencies presented by the calculation. Zero
        means a minimum energy structure, one means a transition state. Larger
        values represent other structures in the potential energy surface.
    """

    d = cc.parser.ccopen(output)
    m = d.parse()

    nimagfreqs = len(m.vibfreqs[m.vibfreqs < 0])

    return m.freeenergy, m.enthalpy, m.entropy, nimagfreqs


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
    molecularity : int
        The molecularity of the reaction step, defined as the number of quantum
        chemistry log files that are taken into account in this reaction step.
    nimagfreqs : int
        The number of imaginary frequencies presented by the calculation. Zero
        means a minimum energy structure, one means a transition state. Larger
        values represent other structures in the potential energy surface.
    """

    freeenergy, enthalpy, entropy = 0., 0., 0.
    name = ""
    molecularity = 0
    nimagfreqs = -1

    if isinstance(step, str):
        freeenergy, enthalpy, entropy, nimagfreqs = _free_energy_comps(step)
        name = step
        molecularity = 1
    elif isinstance(step, list):
        for isomer in step:
            _freeenergy, _enthalpy, _entropy, _name, _molecularity, \
                _nimagfreqs = _parse_step(isomer)

            if freeenergy > _freeenergy:
                freeenergy = _freeenergy
                enthalpy, entropy = _enthalpy, _entropy
                name = _name
                molecularity = _molecularity
                nimagfreqs = _nimagfreqs
    elif isinstance(step, tuple):
        nimagfreqs += 1

        for molecule in step:
            _freeenergy, _enthalpy, _entropy, _name, _molecularity, \
                _nimagfreqs = _parse_step(molecule)

            freeenergy += _freeenergy
            enthalpy += _enthalpy
            entropy += _entropy
            name = "{:s}+{:s}".format(name, _name)
            molecularity += _molecularity
            nimagfreqs += _nimagfreqs

        name = name[1:]
    else:
        raise TypeError("step type should be str, list or tuple")

    return freeenergy, enthalpy, entropy, name, molecularity, nimagfreqs


def equilibrium_constant(barrier, temperature):
    """
    Calculate the constant for a chemical equilibrium:

    $$K = \exp{- \frac{\Delta G^\circ}{R T}}$$

    where $K$ is the equilibrium constant, $\Delta G^\circ$ is the reaction
    free energy, $T$ is the absolute temperature and $R$ is the gas constant.
    """
    return np.exp(-joulespermol * barrier/(gas_constant * temperature))


def _rate_constant_eyring(barrier, molecularity, temperature, pressure,
                          concentration):
    """
    Calculate the reaction rate constant for a chemical process according to
    the Eyring-Evans-Polanyi equation:

    $$k = \frac{k_B T}{h c_0} \exp{- \frac{\Delta G^\neq}{R T}}$$

    where $k$ is the reaction rate constant, $\Delta G^\neq$ is the Gibbs
    energy of activation, $c_0$ is the concentration of the reactant, $k_B$ is
    Boltzmann's constant, $h$ is Planck's constant, $T$ is the absolute
    temperature and $R$ is the gas constant.

    For information on the use of this equation, see for example
    <http://gaussian.com/thermo/>.

    Parameters
    ----------
    barrier : float
        Gibbs energy of activation ($\Delta G^\neq$) for the chemical process,
        in hartree/particle.
    molecularity : int
        The molecularity of the reaction step, defined as the number of quantum
        chemistry log files that are taken into account in this reaction step.
        The use of this value has not been implemented yet.
    temperature : float
        Absolute temperature at which reaction rate constant should be
        calculated.
    pressure : float
        Pressure used to calculate `concentration`, assuming ideal gas
        behaviour. This is ignored if `concentration` is given (see below).
    concentration : float
        Concentration of the reactant in mol/m^3. If given, this is used in
        favour of `pressure` (see above), which is then ignored.

    Returns
    -------
    float
        Reaction rate constant for the particular process according to the
        Eyring-Evans-Polanyi equation. Units are in [m^3/mol]^(n - 1)/s, where
        n is the reaction step molecularity.
    """

    if concentration is None:
        concentration = pressure / (gas_constant * temperature)

    pre_factor = Boltzmann * temperature / (Planck * concentration)
    rate = pre_factor * equilibrium_constant(barrier, temperature)

    return rate


def rate_constant(barrier, molecularity=1, temperature=298.15, pressure=1e5,
                  concentration=None, method="eyring"):
    """
    Calculate the reaction rate constant for a chemical process.

    Parameters
    ----------
    barrier : float
        Gibbs energy of activation ($\Delta G^\neq$) for the chemical process,
        in hartree/particle.
    molecularity : int, optional
        The molecularity of the reaction step, defined as the number of quantum
        chemistry log files that are taken into account in this reaction step.
        The use of this value has not been implemented yet.
    temperature : float, optional
        Absolute temperature at which reaction rate constant should be
        calculated.
    pressure : float, optional
        Pressure used to calculate `concentration`, assuming ideal gas
        behaviour. This is ignored if `concentration` is given (see below).
    concentration : float, optional
        Concentration of the reactant in mol/m^3. If given, this is used in
        favour of `pressure` (see above), which is then ignored.
    method : {"eyring"}, optional
        Equation used for the calculation of the reaction rate constant.

    Returns
    -------
    float
        Reaction rate constant for the particular process. Units are in
        [m^3/mol]^(n - 1)/s, where n is the reaction step molecularity.
    """

    if method == "eyring":
        rate = _rate_constant_eyring(barrier, molecularity, temperature,
                                     pressure, concentration)
    else:
        raise ValueError("unknown equation {:s}".format(method))

    return rate


def mechanism(steps, G=None, temperature=298.15, pressure=1e5,
              concentration=None, method="eyring"):
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
    temperature : float, optional
        Absolute temperature at which reaction rate constants should be
        calculated.
    pressure : float, optional
        Pressure used to calculate `concentration`s, assuming ideal gas
        behaviour. This is ignored if `concentration` is given (see below).
    concentration : float, optional
        Concentration of each reactant in mol/m^3. If given, this is used in
        favour of `pressure` (see above), which is then ignored.
    method : {"eyring"}, optional
        Equation used for the calculation of the reaction rate constants.

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
    last_freeenergy = None
    last_nimagfreqs = None
    for step in steps:
        freeenergy, enthalpy, entropy, name, molecularity, nimagfreqs = \
            _parse_step(step)

        G.add_node(name, freeenergy=freeenergy, enthalpy=enthalpy,
                   entropy=entropy)

        if last_name is not None:
            barrier = freeenergy - last_freeenergy
            if nimagfreqs == 1 and last_nimagfreqs == 0:
                k = rate_constant(barrier, molecularity, temperature, pressure,
                                  concentration, method)
                G.add_edge(last_name, name, k=k)
            elif nimagfreqs == 0 and last_nimagfreqs == 0:
                K = equilibrium_constant(barrier, temperature)
                G.add_edge(last_name, name, K=K)
            else:
                G.add_edge(last_name, name)

        last_name = name
        last_freeenergy = freeenergy
        last_nimagfreqs = nimagfreqs

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

    all_simple_paths = nx.all_simple_paths(G, source, target)
    for path in all_simple_paths:
        path_freeenergies = []
        for step in path:
            path_freeenergies.append(freeenergies_dict[step])

        path_freeenergies = np.array(path_freeenergies)

        transitions = list(zip(path[:-1], path[1:]))

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
        side_step = np.array([.63e-3, -.10])

        above_step *= step_width * mpl.rcParams["font.size"]
        below_step *= step_width * mpl.rcParams["font.size"]
        side_step *= step_width * mpl.rcParams["font.size"]

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

        for i, (xx, yy) in enumerate(zip(transition_x_ranges,
                                         transition_y_ranges)):
            plt.plot(xx, yy, transition_pattern)
            pos = np.array([np.average(xx), np.average(yy)])

            if "k" in G.edges[transitions[i]]:
                note = "k={:.2g}".format(G.edges[transitions[i]]["k"])
                adjust = np.array([len(note), 1.])
                plt.annotate(note, pos + side_step * adjust)
            elif "K" in G.edges[transitions[i]]:
                note = "K={:.2g}".format(G.edges[transitions[i]]["K"])
                adjust = np.array([len(note), 1.])
                plt.annotate(note, pos + side_step * adjust)

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
