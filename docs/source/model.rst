Model
=====

GPSeer uses three models that can be applied in serial to predict unmeasured
phenotypes in genotype-phenotype maps. The linear model is always used.
The classifier and spline are both optional.  The classifier is always applied
first, followed by the spline.

Classifier
----------

Logistic classifier that determines whether or not a genotype is below or above
an assay detection threshold. Each mutation can make an additive contribution
to the classifier.  This uses the `sklearn logistic regression model <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_,
as wrapped by the `epistasis package <https://epistasis.readthedocs.io/gallery/plot_logistic_regression.html>`_.

Spline
------

A nonlinear spline to account for nonlinear mapping between the additive effects
of mutations and the observed phenotypes.  This uses the `scipy UniverateSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html>`_
as wrapped by the `epistasis package <https://epistasis.readthedocs.io/gallery/plot_nonlinear_regression.html>`_.

Linear Model
------------

A linear model that describes the effects of mutations as additive perturbations
to phentoype.  It is possible to include pairwise and higher-ordered epistatic
coefficients, however this is not recommended.  This model uses the
`epistasis package <https://epistasis.readthedocs.io/pages/models.html#epistasislinearregression>`_.

Reading
-------

+ Citation coming soon
+ `Sailer & Harms (2017) Genetics <https://www.genetics.org/content/205/3/1079.abstract>`_.
+ `Otwinowski et al. (2018) PNAS <https://www.pnas.org/content/115/32/E7550>`_.
