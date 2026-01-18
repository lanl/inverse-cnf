#!/bin/sh


##################################################
#          DO NOT MODIFY ABOVE THIS LINE         #
##################################################

INVERSE_CNF_PROJECT_DIR="/absolute/path/to/inverse-cnf"

INVERSE_CNF_OUTPUT_DIR="/absolute/path/to/output_directory"

##################################################
#          DO NOT MODIFY BELOW THIS LINE         #
##################################################


if [ -n "$BASH_SOURCE" ]; then
    script_path="${BASH_SOURCE[0]}"
elif [ -n "${(%):-%N}" ]; then
    script_path="${(%):-%N}"
else
    echo "Please run this script in BASH or ZSH shell"
    return 1
fi


# helper functions
ERROR_OCCURRED=0
death() {
    printf 'ERROR: %s\n' "$*" >&2
    ERROR_OCCURRED=1
    return 1
}

print_div() { 
    printf '%*s\n' 100 '' | tr ' ' '-'; 
}

EXPECTED_PROJECT_DIR="inverse-cnf"
# validate root paths
validate_paths() {
    case "$INVERSE_CNF_PROJECT_DIR" in /*) ;; *) death "INVERSE_CNF_PROJECT_DIR '$INVERSE_CNF_PROJECT_DIR' is not an absolute path" ;; esac
    [ -d "$INVERSE_CNF_PROJECT_DIR" ] || death "INVERSE_CNF_PROJECT_DIR '$INVERSE_CNF_PROJECT_DIR' does not exist or is not a directory"

    if [ "$(basename "$INVERSE_CNF_PROJECT_DIR")" != "$EXPECTED_PROJECT_DIR" ]; then
        death "INVERSE_CNF_PROJECT_DIR '$INVERSE_CNF_PROJECT_DIR' does not point to project root '$EXPECTED_PROJECT_DIR'"
    fi

    [ "$ERROR_OCCURRED" -eq 0 ] || return 1

    case "$INVERSE_CNF_OUTPUT_DIR" in /*) ;; *) death "INVERSE_CNF_OUTPUT_DIR '$INVERSE_CNF_OUTPUT_DIR' is not an absolute path" ;; esac
    [ -d "$INVERSE_CNF_OUTPUT_DIR" ] || death "INVERSE_CNF_OUTPUT_DIR '$INVERSE_CNF_OUTPUT_DIR' does not exist or is not a directory"

    [ "$ERROR_OCCURRED" -eq 0 ] || return 1
}


# export project paths
export_project_paths() {
    print_div
    printf 'Exporting project paths as environment variables\n\n'
    export INVERSE_CNF_PROJECT_DIR="$INVERSE_CNF_PROJECT_DIR"
    printf 'Exported: INVERSE_CNF_PROJECT_DIR=%s\n' "$INVERSE_CNF_PROJECT_DIR"
    export INVERSE_CNF_OUTPUT_DIR="$INVERSE_CNF_OUTPUT_DIR"
    printf 'Exported: INVERSE_CNF_OUTPUT_DIR=%s\n' "$INVERSE_CNF_OUTPUT_DIR"
}

# validate and create output folders
check_output_folders() {
    output_folder_list=$(printf '%s\n' \
        "$INVERSE_CNF_OUTPUT_DIR/datasets" \
        "$INVERSE_CNF_OUTPUT_DIR/subsets" \
        "$INVERSE_CNF_OUTPUT_DIR/databases" \
        "$INVERSE_CNF_OUTPUT_DIR/experiments")

    [ -n "${ZSH_VERSION-}" ] && output_folder_list=("${(f)output_folder_list}")

    print_div
    printf 'Checking if project output folders exist\n\n'

    temp_loop_path=""
    for temp_loop_path in $output_folder_list; do
        if [ ! -d "$temp_loop_path" ]; then
            mkdir -p "$temp_loop_path" || mkdir "$temp_loop_path" || terminate "Could not create $temp_loop_path"
            printf 'Created: %s\n' "$temp_loop_path"
        else
            printf 'Verified: %s\n' "$temp_loop_path"
        fi
    done
}


# validate python path 
check_python_path() {
    python_path_list=$(printf '%s\n' \
        "$INVERSE_CNF_PROJECT_DIR" \
        "$INVERSE_CNF_PROJECT_DIR/programs" \
        "$INVERSE_CNF_PROJECT_DIR/simulators")

    [ -n "${ZSH_VERSION-}" ] && python_path_list=("${(f)python_path_list}")

    print_div
    printf 'Checking if project paths are present in PYTHONPATH\n\n'

    is_updated=0
    temp_pythonpath=$PYTHONPATH
    combined_pythonpath=""
    for temp_loop_path in $python_path_list; do
        if [[ ":$temp_pythonpath:" == *":$temp_loop_path:"* ]]; 
        then
            printf 'Verified: %s\n' "$temp_loop_path"
        else
            combined_pythonpath="${combined_pythonpath:+$combined_pythonpath:}$temp_loop_path"
            printf 'Appended: %s\n' "$temp_loop_path"
            is_updated=1
        fi
    done

    if [ "$is_updated" -eq 1 ];then
        export PYTHONPATH="${combined_pythonpath:+:$combined_pythonpath:}$temp_pythonpath"
    fi
    printf '\nPYTHONPATH: %s\n' "$PYTHONPATH"
}


# run each stage and check for errors
validate_paths || { print_div; return 1 2>/dev/null; } || :

export_project_paths

check_python_path

check_output_folders || { print_div; return 1 2>/dev/null; } || :