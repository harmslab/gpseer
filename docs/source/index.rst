.. gpseer documentation master file, created by
   sphinx-quickstart on Sat Sep 14 06:17:16 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
GPSeer
======

*Simple software for inferring missing data in sparsely measured genotype-phenotype maps*

Background
==========

Experimentally characterizing genotype-phenotype maps can be very challenging because the size of a map expands exponentially as the number of mutations increases.  For example, a map with four mutational sites, each existing in one of two states, includes 16 genotypes (:math:`2^{4}`).  By contrast, a map with 15 mutational sites consists of 32,768 genotypes (:math:`2^{15}`).  Exhaustive characterization of the phenotypes in a map is often infeasible, particularly for phenotypes that are difficult to characterize by high-throughput methods.

To address this shortfall, we have developed a straightforward approach to infer the missing phenotypes from an incomplete genotype-phenotype map, with well-characterized uncertainty in our predictions.  Such knowledge allows robust and statistically-informed analyses of features of the map, such as knowledge of possible evolutionary trajectories.


.. toctree::
    :maxdepth: 1
    :hidden:

    install
    basic
    io
    interface
