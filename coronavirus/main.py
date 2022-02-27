# -*- encoding: utf-8 -*-
# Date: 31/Mar/2020
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""
Description: COVID-19 visualization and analysis
"""

from json_update import update_json
from plot_cases import init_folder, plotCountriesFromOurWorld
from plot_vaccinations import downloadOurWorldData, startPlotVaccination


def main():
    init_folder()
    downloadOurWorldData()
    plotCountriesFromOurWorld()
    startPlotVaccination()
    update_json()


if __name__ == '__main__':
    main()
