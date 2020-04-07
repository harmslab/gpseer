import pandas as pd
from gpmap import GenotypePhenotypeMap


def read_input_file(input_file_name):
    """Read the input file for GPSeer.

    This should be a CSV file with the following columns:
    genotypes, phenotypes, n_replicates, stdeviations
    """
    df = pd.read_csv(input_file_name, index_col=0)
    required_columns = ["genotypes","phenotypes"]
    for c in required_columns:
        try:
            df[c]
        except AttributeError:
            err = "input file ({}) must contain a column labeled '{}'".format(input_file, c)
            return
    return df


def _raise_line_err(msg,line):
    err = "\n\n{}\n\n".format(msg)
    err += "Line:\n\n{}\n\n".format(line.strip())
    raise ValueError(err)


def read_genotype_file(wildtype, genotype_file_name):
    """Read a file with a list of genotypes to predict.
    """
    genotype_size = len(wildtype)

    out_genotypes = []
    with open(genotypes_file_name) as f:
        for line in f.readlines():
            genotype = line.strip()

            # Skip blank lines and # comments
            if genotype == "" or genotype.startswith("#"):
                continue

            # Look for line with more than one genotype
            if len(genotype.split()) > 1:
                _raise_line_err("Mangled line. More than one genotype?",line)

            # Look for line with incorrect number of sites
            if len(genotype) != genotype_size:
                _raise_line_err("Mangled line. Genotype length does not match {}".format(wildtype),line)

            out_genotypes.append(genotype)

    return out_genotypes


def gpmap_from_gpmap(
    original_gpmap,
    new_genotypes,
    new_phenotypes,
    new_n_replicates=1,
    new_stdeviations=None,
    ):
    """Generate a GenotypePhenotypeMap from another GenotypePhenotypeMap
    with new genotypes, phenotypes, n_replicates, and stdevations.
    """
    gpm = original_gpmap
    return GenotypePhenotypeMap(
        wildtype=gpm.wildtype,
        mutations=gpm.mutations,
        genotypes=new_genotypes,
        phenotypes=new_phenotypes,
        stdeviations=new_stdeviations,
        n_replicates=new_n_replicates
    )


def write_output_file(
    output_file_name,
    out_df
    ):
    out_df.to_csv(output_file_name)
