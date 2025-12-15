#' Run a CloudComPy Script
#'
#' @description
#' This function runs a Python script that uses CloudComPy.
#'
#' @param script_path The path to the Python script to run with CloudComPy.
#' @param check_installation Logical. Whether to verify CloudComPy installation
#' before running. Defaults to TRUE for safety, but should be set to FALSE once
#' the installation was confirmed to be valid to boost performance.
#'
#' @details
#' This function assumes that CloudComPy has been installed via
#' install_cloudcompy(). By default, it performs a check to validate the
#' installation.
#'
#' For proper CloudComPy functionality, ensure install_cloudcompy() has been run
#' before executing this function!
#'
#' @importFrom reticulate source_python
#'
#' @export
#' @examples
#' \dontrun{
#' # Assuming CloudComPy is installed
#' source_ccpy("path/to/your/script.py")
#'
#' # Skip installation check for performance
#' source_ccpy("path/to/your/script.py", check_installation = FALSE)
#' }
#'
source_ccpy <- function(script_path, check_installation = TRUE) {
  if (check_installation && !.check_ccpy_installation()) {
    stop(
      "CloudComPy is not properly installed.",
      "Run install_cloudcompy() first."
    )
  }

  reticulate::source_python(script_path)
}
"C:/Users/alex-work/Desktop/load_cloud_test.py"