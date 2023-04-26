"""
 * swdatatoolkit, a project at the Data Mining Lab
 * (http://dmlab.cs.gsu.edu/) of Georgia State University (http://www.gsu.edu/).
 *
 * Copyright (C) 2022 Georgia State University
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation version 3.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
"""

from enum import Enum


class PatchSize(Enum):
    """
    This contains all possible sizes of a patch that parameters are going to be
    computed for.

    Note: For adding new items to the list, make sure that the input
    images can be divided by the new number. (bImage.getWidth % newItem == 0)

    """
    ONE = 1
    FOUR = 4
    SIXTEEN = 16
    THIRTY_TWO = 32
    SIXTY_FOUR = 64
    ONE_TWENTY_EIGHT = 128
    TWO_FIFTY_SIX = 256
    FIVE_TWELVE = 512
    TEN_TWENTY_FOUR = 1024
    FULL = -1


