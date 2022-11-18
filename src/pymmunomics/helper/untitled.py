def read_sequence_data_files(
    directory_specifications,
    invalid_infixes,
    ntc_infixes,
    useless_datafile_suffixes,
    fileprefix_to_samplename,
    date_to_code,
    column_combination_to_value_keys,
    columns_to_valid_combinations,
    immunodb_data,
    immunodb_cols,
    sequence_cols,
):
    filepath_to_unparsable = {}
    data_chunks = []
    ntc_data_chunks = [] # NTC data should not be input... separate from true input manually; these are for quality control preceding sequence data analysis
    LOGGER.info("Reading raw data...")
    # Make flat file list and provide function for extracting "data-grouping" columns
    for specification in tqdm(directory_specifications):
        filepaths = glob(f"{specification.directorypath}/*")
        for filepath in filepaths:
            unparsable = set()
            result = read_sequence_data_file(
                filepath=filepath,
                read_kwargs=specification.read_kwargs,
                sequencing_date=specification.sequencing_date,
                unparsable_filenames=unparsable,
                invalid_infixes=invalid_infixes,
                ntc_infixes=ntc_infixes,
                useless_datafile_suffixes=useless_datafile_suffixes,
                fileprefix_to_samplename=fileprefix_to_samplename,
                sequence_cols=sequence_cols,
            )
            if result is not None: # What is a None-result?
                data, is_ntc = result
                if is_ntc:
                    ntc_data_chunks.append(data)
                else:
                    data_chunks.append(data)
                filepath_to_unparsable[filepath] = unparsable

    gene_to_has_stopcodon_frameshift = immunodb_data.set_index(
        immunodb_cols["gene_col"],
    )[[immunodb_cols["has_stopcodon_frameshift_col"]]].to_dict()[
        immunodb_cols["has_stopcodon_frameshift_col"]
    ] # seems like this was done already for valid_column_combinations?
    gene_to_has_stopcodon_frameshift["nan"] = "nan"
    time_before = time()
    original_raw_data = concat(data_chunks, ignore_index=True)
    for vj in ["v", "j"]: # annotate via .assign(new_col=df["old_col"].map(dictionary))
        original_raw_data = original_raw_data.assign(
            **{
                sequence_cols[f"{vj}_has_stopcodon_frameshift_col"]: original_raw_data[
                    sequence_cols[f"{vj}gene_col"]
                ].map(lambda gene: gene_to_has_stopcodon_frameshift[gene])
            }
        )
    corrected_raw_data, incorrect_raw_data = correct_idb_types( # Implement as function chain each of which logs what is being corrected
        data=original_raw_data,
        column_combination_to_value_keys=column_combination_to_value_keys,
        sequence_cols=sequence_cols,
    )
    time_between = time()
    LOGGER.info(
        f"Correcting %s took %.2fmin",
        sequence_cols["clone_type_col"],
        (time_between - time_before) / 60,
    )

    final_data = assert_valid_clone_type_corrections(
        data=corrected_raw_data,
        columns_to_valid_combinations=columns_to_valid_combinations,
        sequence_cols=sequence_cols,
    ).pipe(
        lambda df: (
            df.assign(
                **{
                    sequence_cols["patient_col"]: df[sequence_cols["patient_col"]].map(
                        partial(scramble_date, date_to_code=date_to_code)
                    )
                }
            )
        )
    )
    time_after = time()
    LOGGER.info("Validating corrections took %.2fmin", (time_after - time_between) / 60)

    return (
        final_data,
        concat(ntc_data_chunks, ignore_index=True),
        incorrect_raw_data,
        filepath_to_unparsable,
    )



def read_sequence_data_file(
    filepath,
    read_kwargs,
    sequencing_date,
    unparsable_filenames,
    invalid_infixes,
    ntc_infixes,
    useless_datafile_suffixes,
    fileprefix_to_samplename,
    sequence_cols,
):
    filename = filepath.split("/")[-1]
    subtype = parse_subtype(filename) # part of outer concat key
    is_ntc = any(infix in filename for infix in ntc_infixes) # shouldn't be here...
    patient = parse_patient( # This should be done in advance to exclude superfluous files and filenpath components should be converted via outer concat key functions
        filename,
        unparsable_filenames=unparsable_filenames,
        forbidden_substrings=[*invalid_infixes, *ntc_infixes],
        fileprefix_to_samplename=fileprefix_to_samplename,
        useless_datafile_suffixes=useless_datafile_suffixes,
    )
    if any(infix in filename for infix in invalid_infixes): # This should be validated before
        warn(PreprocessingWarning(f"Invalid sample '{filename}' will be ignored"))
    elif filename in unparsable_filenames and not is_ntc: # Also done in advance
        warn(PreprocessingWarning(f"Unparsable filename '{filename}' will be ignored"))
    else:
        data = read_csv(filepath, **read_kwargs).assign(
            **{
                sequence_cols["subtype_col"]: subtype,
                sequence_cols["sequencing_date_col"]: sequencing_date,
                sequence_cols["source_col"]: filepath,
                sequence_cols["patient_col"]: patient,
            }
        )
        return (data.fillna("nan"), is_ntc)