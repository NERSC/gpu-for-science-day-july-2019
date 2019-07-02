#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for GPP
"""

import os
import sys
import glob
import shutil
import argparse
import warnings
import traceback

import pyctest.pyctest as pyctest
import pyctest.pycmake as pycmake
import pyctest.helpers as helpers


def cleanup(path=None, exclude=[]):
    """
    route for cleaning up testing files
    """
    exclude += glob.glob('**/*.cpp', recursive=True)
    exclude += glob.glob('**/Makefile', recursive=True)
    exclude += glob.glob('**/*.cu', recursive=True)
    helpers.Cleanup(path, exclude=exclude)


def configure():

    # set site if set in environ
    if os.environ.get("CTEST_SITE") is not None:
        pyctest.CTEST_SITE = os.environ.get("CTEST_SITE")
    elif os.environ.get("NERSC_HOST") is not None:
        pyctest.CTEST_SITE = os.environ.get("NERSC_HOST")
    elif os.environ.get("HOSTNAME") is not None:
        pyctest.CTEST_SITE = os.environ.get("HOSTNAME")

    # the current directory of this file
    this_dir = os.path.dirname(__file__)

    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(project_name="gpu-for-science-day-july-2019",
                                    source_dir=this_dir,
                                    binary_dir=os.path.join(this_dir, "gpp"),
                                    drop_site="cdash.nersc.gov",
                                    drop_method="https",
                                    submit=True,
                                    ctest_args=["-V"])
    # compiler choices
    compiler_choices = ["intel", "gcc", "cuda", "openacc", "openmp", "kokkos"]

    parser.add_argument("--compiler", type=str, choices=compiler_choices, required=True,
                        help="Select the compiler")
    parser.add_argument("--team", type=str, required=True,
                        help="Specify the team name")
    parser.add_argument("--cleanup", action='store_true', default=False,
                        help="Cleanup of any old pyctest files and exit")
    parser.add_argument("--type", help="Execute either test or benchmark",
                        type=str, default="benchmark", choices=["benchmark", "test"])
    parser.add_argument("-d", "--test-dir", help="Set the testing directory",
                        default=os.path.join(this_dir, "gpp"), type=str)
    parser.add_argument("--post-cleanup", type=bool, default=True,
                        help="Cleanup pyctest files after execution")

    args = parser.parse_args()

    if args.cleanup:
        cleanup(pyctest.BINARY_DIRECTORY)
        sys.exit(0)

    if os.path.exists(args.test_dir):
        pyctest.BINARY_DIR = os.path.abspath(args.test_dir)

    git_exe = helpers.FindExePath("git")
    if git_exe is not None:
        pyctest.UPDATE_COMMAND = "{}".format(git_exe)
        pyctest.set("CTEST_UPDATE_TYPE", "git")

    return args


def run_pyctest():
    '''
    Configure PyCTest and execute
    '''
    # run argparse, checkout source, copy over files
    args = configure()

    # make sure binary directory is clean
    cmd = pyctest.command(["make", "clean"])
    cmd.SetErrorQuiet(True)
    cmd.SetWorkingDirectory(pyctest.BINARY_DIRECTORY)
    cmd.Execute()

    # make sure there is not an existing (old) Testing directory
    if os.path.exists(os.path.join(pyctest.BINARY_DIR, "Testing")):
        shutil.rmtree(os.path.join(pyctest.BINARY_DIR, "Testing"))

    #   BUILD_NAME
    pyctest.BUILD_NAME = "{}-{}".format(args.team, args.compiler)
    pyctest.BUILD_COMMAND = "make COMP={}".format(args.compiler)

    # properties
    bench_props = {"TIMEOUT": "10800",
                   "WORKING_DIRECTORY": os.path.join(pyctest.BINARY_DIRECTORY)}

    # create test
    pyctest.test("benchmark", ["srun", "./gpp.ex", "{}".format(args.type)],
                 properties=bench_props)

    print('Running PyCTest:\n\n\t{}\n\n'.format(pyctest.BUILD_NAME))

    pyctest.run()

    # remove these files
    files = ["CTestConfig.cmake", "CTestCustom.cmake",
             "CTestTestfile.cmake", "Init.cmake", "Stages.cmake", "Utilities.cmake"]

    try:
        if args.post_cleanup:
            for f in files:
                if os.path.exists(os.path.join(pyctest.BINARY_DIR, f)):
                    os.remove(os.path.join(pyctest.BINARY_DIR, f))
            shutil.rmtree(os.path.join(pyctest.BINARY_DIR, "Testing"))
    except:
        pass


if __name__ == "__main__":
    try:
        run_pyctest()
    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)
    sys.exit(0)
