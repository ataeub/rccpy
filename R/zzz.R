#' Check if CloudComPy is properly installed.
#'
#' @return Logical indicating if CloudComPy installation is valid
#' @keywords internal
.check_ccpy_installation <- function() {
  cli::cli_progress_step("Checking CloudComPy installation...")

  pkg_dir <- tools::R_user_dir("ccpypr")

  cli::cli_progress_step("Verifying directories...")

  # Check if required directories exist
  miniconda_exists <- dir.exists(file.path(pkg_dir, "r-miniconda"))
  cloudcompy_exists <- dir.exists(file.path(pkg_dir, "CloudComPy310"))

  if (!miniconda_exists || !cloudcompy_exists) {
    cli::cli_progress_done()
    return(FALSE)
  }

  # Verify that the environment works
  cli::cli_progress_step("Activating environment...")

  suppressWarnings({
    tryCatch(
      {
        miniconda_dir <- file.path(pkg_dir, "r-miniconda")
        miniconda_exe <- file.path(miniconda_dir, "condabin/conda.bat")
        condaenv_dir <- file.path(
          miniconda_dir,
          "envs/CloudComPy310/python.exe"
        )

        reticulate::use_condaenv(
          conda = miniconda_exe,
          condaenv = condaenv_dir,
          required = TRUE
        )

        # Try a simple Python import
        reticulate::py_run_string("import sys")

        cli::cli_progress_step("Testing Python imports...")

        # Try if CloudComPy and cloudpy can be imported
        python_path <- system.file("python", package = "rccpy")
        path_mngmn <- reticulate::import_from_path(
          "path_management",
          python_path,
          delay_load = TRUE
        )
        original_paths <- path_mngmn$get_sys_paths()
        cloudcompy_dir <- file.path(pkg_dir, "CloudComPy310")
        path_mngmn$set_ccpy_paths(cloudcompy_dir)
        reticulate::source_python(system.file(
          "python/import_ccpy.py",
          package = "rccpy"
        ))
        reticulate::source_python(system.file(
          "python/cloudcompype/cloudcompype.py",
          package = "rccpy"
        ))

        # We do already reset the paths here. However for some reason CloudComPy
        # does function later on, without these paths apparently.
        # Idk why, it just works...
        # Just keep this in mind, in case this becomes an issue later.
        path_mngmn$reset_paths(original_paths)

        cli::cli_progress_done()
        return(TRUE)
      },
      error = function(e) {
        cli::cli_alert_danger(e$message)
        return(FALSE)
      }
    )
  })
}

.onAttach <- function(libname, pkgname) {
  version <- utils::packageVersion(pkgname)
  packageStartupMessage("\nThis is ", pkgname, " version ", version, ".\n")

  # Check CloudComPy installation
  if (!.check_ccpy_installation()) {
    packageStartupMessage(
      "\nNOTE: CloudComPy does not appear to be installed properly."
    )
    packageStartupMessage(
      "Run install_cloudcompy() to set it up."
    )
    packageStartupMessage(
      "See ?install_cloudcompy for details.\n"
    )
  }
}
