"""
 * NRT-HARP-Data-Processor, a project at the Data Mining Lab
 * (http://dmlab.cs.gsu.edu/) of Georgia State University (http://www.gsu.edu/).
 *
 * Copyright (C) 2020 Georgia State University
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

class DataType(Enum):
    """
    This contains all possible datatypes that can be downloaded for HARP data series.
    """
    NRT = 'NRT'
    DEFINITIVE = 'DEFINITIVE'
    
class QueryFilter(Enum):
    """
    This contains the query strings for all possible datatypes that can be downloaded for HARP data series.
    """
    NRT = 'hmi.sharp_cea_720s_nrt'
    DEFINITIVE = 'hmi.sharp_cea_720s'