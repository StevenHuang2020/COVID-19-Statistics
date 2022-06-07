# -*- encoding: utf-8 -*-
# Date: 31/Mar/2020
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""
Description: COVID-19 visualization and analysis
"""
from argparse import ArgumentParser
from json_update import update_json
from plot_cases import init_folder, plotCountriesFromOurWorld
from plot_vaccinations import downloadOurWorldData, startPlotVaccination


def create_params():
    parser = ArgumentParser(description='COVID-19 visualization and analysis')
    parser.add_argument('-s', '--show', action='store_false', default=True,
                        help="show plots or not")

    return parser.parse_args()


def main():
    options = create_params()
    # print("show=", options.show)
    show = options.show

    init_folder()
    downloadOurWorldData()
    plotCountriesFromOurWorld(show=show)
    startPlotVaccination(show=show)
    update_json()


if __name__ == '__main__':
    main()
