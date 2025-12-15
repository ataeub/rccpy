#' Install CloudComPy from a bin
#'
#' @description
#' Unpacks the CloudComPy bin into your user directory, creates a miniconda
#' instance there and creates a conda environment for CloudComPy.
#' Currently only a single version of ClouComPy is supported (20240613).
#' Download it from here:
#' \url{https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/
#' 3-cloudcompy-binaries/5-windows-cloudcompy-binaries/
#' 106-cloudcompy310-20240613>}
#'
#' @param cloudcompy The path to the CloudComPy archive
#'
#' @importFrom archive archive_extract
#' @importFrom reticulate install_miniconda conda_binary conda_create
#' @importFrom reticulate conda_install
#'
#' @export
#'
install_cloudcompy <- function(cloudcompy) {
  pkg_dir <- tools::R_user_dir("rccpy")
  
  conda_tos_base <- paste0(
    "tos accept --override-channels --channel ",
    "https://repo.anaconda.com/pkgs/"
  )
  conda_tos_main <- paste0(conda_tos_base, "main")
  conda_tos_r <- paste0(conda_tos_base, "r")
  conda_tos_msys2 <- paste0(conda_tos_base, "msys2")

  miniconda_dir <- file.path(pkg_dir, "r-miniconda")
  miniconda_exe <- file.path(pkg_dir, "r-miniconda/condabin/conda.bat")

  archive::archive_extract(cloudcompy,
    dir = pkg_dir
  )

  tryCatch(
    {
      reticulate::install_miniconda(miniconda_dir, update = FALSE)
    },
    error = function(e) {
      print(e)
      if (grepl("[exit code 2]", e)) {
        tmp_dir <- tempdir()
        miniconda_exe <- list.files(tmp_dir,
          full.names = T,
          pattern = "*.exe"
        )
        file.remove(miniconda_exe)
      }
      if (grepl("tos", e)) {
        tos_accept <- FALSE
        while (isFALSE(tos_accept)) {
          input <- utils::menu(
            choices = c("OK", "NOPE", "GET ME OUT OF HERE!"),
            title = paste0(
              "To continue you have to accept the tos of the",
              "conda channels: main, r and msys2"
            )
          )
          if (input == 1) {
            tos_accept <- TRUE
          } else if (input == 3) {
            break
          }
          if (isTRUE(tos_accept)) {
            system(
              paste(
                reticulate::conda_binary(miniconda_exe),
                conda_tos_main
              )
            )
            system(
              paste(
                reticulate::conda_binary(miniconda_exe),
                conda_tos_r
              )
            )
            system(
              paste(
                reticulate::conda_binary(miniconda_exe),
                conda_tos_msys2
              )
            )
            reticulate::install_miniconda(miniconda_dir)
          }
        }
      }
    }
  )

  conda_env <- system.file("environment.yml", package = "rccpy")
  reticulate::conda_create(
    conda = miniconda_exe,
    environment = conda_env
  )
}
