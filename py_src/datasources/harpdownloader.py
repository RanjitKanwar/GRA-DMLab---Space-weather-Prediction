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

import os
import re
from logging import Logger
from typing import List, Dict

import drms
import time
import glob
import socket
import tarfile
import datetime
import requests
import traceback
import urllib.request
import tempfile
from . import DataType,QueryFilter


class HARPDownloader:
    """
    This is a base abstract class for download HARP Data Series.
    """
      
    def __init__(self, drms_client_email: str, logger: Logger, data_directory: str = tempfile.mkdtemp(prefix="harps_data_",suffix=datetime.datetime.now().strftime('_%d%m%Y_%H%M%S')), datatype: str = 'NRT', filetypes: list = ['magnetogram']):
        """
        Constructor
        :param filetypes: List of filetypes to process e.g magnetogram,bitmap,etc
        :param datatype: :py:class:`datasources.DataType` String of datatype to be processed. e.g NRT(for near realtime), DEFINITIVE(for definitive)
        :param drms_client_email: Registered Email Address on JSOC.
        :data_directory: Download Directory
        :Logger:
        """
        
        self._qfilter = dict((i.name,i.value) for i in QueryFilter)
                               
        self._data_directory = self.__validate_data_directory(data_directory)
        self._datatype = self.__validate_datatype(datatype)
        self._filetypes = self.__validate_filetypes(filetypes,datatype)
        
        self._drms_client_name = drms_client_email
        self._logger = logger
        socket.setdefaulttimeout(60)
                             
    def __validate_datatype(self,datatype):
        if datatype not in [str(member.value) for member in DataType]:
            raise ValueError(f"'{datatype}' is not a valid datatype. Should be one of {','.join([member.value for member in DataType])}")
        return datatype
                             
    def __validate_filetypes(self,filetypes,datatype):
        c = drms.Client(verbose=False)
        allowed_filetypes = c.info(self._qfilter[datatype]).segments.index.values.tolist()
        for filetype in filetypes:
            if filetype not in allowed_filetypes:
                raise ValueError(f"'{filetype}' is not Valid in filetypes. Should be any of {','.join(allowed_filetypes)}")
        return filetypes
                                
    def __validate_data_directory(self,data_directory):
        if not os.path.exists(data_directory):
            raise ValueError(f"Directory '{data_directory}' does not exist")
        return data_directory
                                 
    def __filter_online(self,harp_records:dict) -> Dict:
        """
        Cleans the list of online records to remove those that have already been downloaded. It is expected that each
        HARP number in the dictionary has a list of online records to download into a directory that is named after
        the HARP number being processed.

        :param harp_records: Dictionary of HARP numbers to process with each entry being a list of file times to process
        :return: Cleaned dictionary of online records for each harp number
        """


        answer = {}
        for sNum, recs in harp_records.items():
            tracker = {}
            save_dir = os.path.join(os.path.join(self._data_directory, str(sNum)),self._datatype)

            if not os.path.exists(save_dir):
                answer[sNum] = recs
            else:
                self.__check_for_tar(save_dir)            
                for filetype in self._filetypes:

                    files = []
                    files.extend(glob.glob(save_dir + f"/*{filetype}.fits"))

                    seen = []
                    for file in files:
                        dots = list(self.__find_all(file, '.'))
                        if len(dots) > 2:
                            yr_pos = dots[2] + 1
                            yr = file[yr_pos:yr_pos + 4]
                            mo_pos = yr_pos + 4
                            mo = file[mo_pos:mo_pos + 2]
                            dy_pos = mo_pos + 2
                            dy = file[dy_pos:dy_pos + 2]
                            hr_pos = dy_pos + 3
                            hr = file[hr_pos:hr_pos + 2]
                            mi_pos = hr_pos + 2
                            mi = file[mi_pos:mi_pos + 2]
                            sec_pos = mi_pos + 2
                            sec = file[sec_pos:sec_pos + 2]
                            rec_time = "{0}.{1}.{2}_{3}:{4}:{5}_TAI".format(yr, mo, dy, hr, mi, sec)
                            if rec_time in recs:
                                seen.append(rec_time)

                    tracker[filetype] = list(set(seen))

                all_filetypes = []
                for record in tracker:
                    all_filetypes.append(tracker[record])
                all_filetypes = [item for sublist in all_filetypes for item in sublist]
                
                for rec in recs.copy():
                    if all_filetypes.count(rec) == len(filetypes):
                        recs.remove(rec)
                if len(recs) > 0:
                    answer[sNum] = recs

        return answer


    @staticmethod
    def __find_all(a_str: str, sub: str):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub)

            
    def __check_for_tar(self, save_dir) -> bool:
        """
        Method checks for and unzips tar files within a specified directory.  It will also remove some of the additional
        files that are sometimes included in the tar file as well as partial downloads and the tar file it self.

        :param save_dir: The directory to search for tar files to unzip

        :return: True if a tar file was found to unzip, false otherwise.
        """

        found_files = False
        files = []
        files.extend(glob.glob(save_dir + "/*.tar"))

        for fname in files:
            if fname.endswith("tar"):
                try:
                    tar = tarfile.open(fname, "r:")
                    tar.extractall(save_dir)
                    tar.close()
                    found_files = True
                except Exception as e:
                    self._logger.error('Check for Tar error: %s', str(e))
                    self._logger.debug('Traceback: %s', traceback.format_exc())
                os.remove(fname)

        types = ('*.txt', '*.html', '*.json', '*.drmsrun', '*.qsub', '*.tar.*')
        files = []
        for ftype in types:
            files.extend(glob.glob(save_dir + "/" + ftype))

        for f in files:
            os.remove(f)

        return found_files
            
            
    def __download_online_records(self,harp_records:dict) -> List[int]:
        """
        Downloads the records for each HARP in the dictionary that is passed. It is expected that each HARP number
        in the dictionary has a list of online records to download into a directory that is named after the HARP
        number being processed.

        :param harp_records: Dictionary of HARP numbers to process with each entry being a list of file times to process
        :return: A list of HARP numbers that were successfully downloaded.
        """
        processed_harp_nums = []
        part_size = 100
        for sNum, recs in harp_records.items():
            self._logger.info("SharpNum {0}, Records {1}".format(sNum, len(recs)))

            save_dir = os.path.join(os.path.join(self._data_directory, str(sNum)),self._datatype)
            has_complete = False
            attempt_count = 0
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            while attempt_count < 5 and not has_complete:
                try:
                    self._logger.info('Total records found: %s - HARP: %s', len(recs), sNum)
                    if len(recs) > 0:
                        c = drms.Client(verbose=False)
                        num_partitions = len(recs) // part_size

                        for i in range(num_partitions + 1):
                            ts_tmp = ''
                            if i < num_partitions:
                                for j in range(0, part_size):
                                    ts_tmp += '{0}, '.format(recs[j + i * part_size])
                            else:
                                j = i * part_size
                                while j < len(recs):
                                    ts_tmp += '{0}, '.format(recs[j])
                                    j = j + 1

                            q = f"{self._qfilter[self._datatype]}[{sNum}][{ts_tmp}]"
        
                            k = c.query(q, key='HARPNUM, T_REC')
                            self._logger.debug("Found %s records for query on HARP:%s", len(k), sNum)
                            max_req = 25000

                            self._logger.debug("Starting Download Harp:%s, Partition:%s", sNum, i)

                            r = c.export(
                                q + '{'+ ','.join(self._filetypes) +'}',
                                method='ftp-tar', protocol='fits', n=max_req, email=self._drms_client_name)

                            self._logger.debug('Starting download...')
                            time.sleep(10)

                            try_cnt = 0
                            done = False
                            while try_cnt < 10 and not done:
                                while (not r.has_finished()) and (not r.has_failed()):
                                    r.wait(sleep=30, timeout=120)

                                tarurl = r.urls['url'][0]
                                filename = r.urls['filename'][0]
                                save_loc = os.path.join(save_dir, filename)
                                try:
                                    # make request and set timeout for no data to 120 seconds and enable streaming
                                    with urllib.request.urlopen(tarurl, timeout=120) as resp:
                                        # Open output file and make sure we write in binary mode
                                        with open(save_loc + '.part', 'wb') as fh:
                                            # walk through the request response in chunks of 1MiB
                                            while True:
                                                chunk = resp.read(1024 * 1024)
                                                if not chunk:
                                                    break
                                                # Write the chunk to the file
                                                fh.write(chunk)

                                    # Rename the file
                                    os.rename(save_loc + '.part', save_loc)

                                    done = True
                                    self._logger.info("Data Chunk Downloaded - HARP:%s, Partition:%s", sNum, i)
                                    if self.__check_for_tar(save_dir):
                                        processed_harp_nums.append(sNum)
                                except Exception as e:
                                    self._logger.error('Attempt:%s. - Download Chunk Error: %s - HARP:%s',
                                                       attempt_count,
                                                       str(e), sNum)
                                    self._logger.debug('Download Chunk Error Traceback: %s - HARP:%s',
                                                       traceback.format_exc(), sNum)
                                    time.sleep(10)
                                    try_cnt += 1
                        has_complete = True
                    else:
                        self._logger.info("No records found - HARP:%s", sNum)
                        None
                    has_complete = True
                except Exception as e:
                    self._logger.error('Attempt:%s - Download Chunk Error: %s - HARP:%s', attempt_count, str(e), sNum)
                    self._logger.debug('Download Chunk Error Traceback: %s - HARP:%s', traceback.format_exc(), sNum)
                    time.sleep(10)
                    attempt_count += 1
        response = dict()
        response['data_directory'] = self._data_directory
        response['HarpNum'] = list(set(processed_harp_nums))
        return response           

            
    def get_list_of_online_by_time_range(self,start_time: datetime, end_time: datetime) -> dict:
        """
        Gets a json object that contains a dictionary of all the online records for the NRT SHARP dataset. The
        dictionary contains an entry for each HARP that is found in the range, and each entry is a list of files
        available for that harp number in the the given date range.

        :param start_time: The start time of the range
        :param end_time: The end time of the range
        :param datatype: String of datatype to be processed. e.g NRT(for near realtime), DEFINITIVE(for definitive)
        :return: Dictionary with a list of online file times to download from JSOC for each HARP number found in range.
        """

        url = 'http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info'

        start_time_format = start_time.strftime("%Y.%m.%d_%H:%M:%S")
        end_time_format = end_time.strftime("%Y.%m.%d_%H:%M:%S")

        answer = {}

        time_range = start_time_format + "-" + end_time_format
        
        ds_param = f"{self._qfilter[self._datatype]}[][{time_range}]"
        t_rec_pattern = f"{self._qfilter[self._datatype]}\[(.*?)\]\[(.*?)\]"

        params = dict(
            ds=ds_param,
            op='rs_list',
            key='*online*,*spec*',
            f=1
        )

        try_count = 0
        done = False
        while try_count < 5 and not done:
            try:
                resp = requests.get(url=url, params=params, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()

                    for key, values in data.items():
                        if key == "recset":
                            for val in values:
                                if val["online"] == "Y":
                                    t_record = re.search(t_rec_pattern, val["spec"])
                                    if t_record.group(1) in answer:
                                        answer[t_record.group(1)].append(t_record.group(2))
                                    else:
                                        answer[t_record.group(1)] = [t_record.group(2)]

                    self._logger.info("Online Records JSON Downloaded - Time Range:%s", time_range)
                    done = True
                else:
                    self._logger.critical("Online Records HTTP request Failed: %s - Time Range:%s", resp.status_code,
                                          time_range)
                    done = False
                    try_count += 1

            except Exception as e:
                self._logger.critical("Online Records JSON Download Failed. Error:%s - Time Range:%s", str(e),
                                      time_range)
                try_count += 1

        if done:
            return answer
        else:
            return None

        

    def get_list_of_online_by_sharpNum(self,sharpNum) -> dict:
        """
        Gets a json object that contains a dictionary of all the online records for the NRT SHARP dataset. The
        dictionary contains an entry for each HARP that is found in the range, and each entry is a list of files
        available for that harp number in the the given date range.

        :param sharpNum: The Harp Num to downloading data for
        :param datatype: String of datatype to be processed. e.g NRT(for near realtime), DEFINITIVE(for definitive)
        :return: Dictionary with a list of online file times to download from JSOC for each HARP number found in range.
        """

        url = 'http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info'

        answer = {}
            
        ds_param = f"{self._qfilter[self._datatype]}[{sharpNum}][]"
        t_rec_pattern = f"{self._qfilter[self._datatype]}\[(.*?)\]\[(.*?)\]"

        params = dict(
            ds=ds_param,
            op='rs_list',
            key='*online*,*spec*',
            f=1
        )

        try_count = 0
        done = False
        while try_count < 5 and not done:
            try:
                resp = requests.get(url=url, params=params, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()

                    for key, values in data.items():
                        if key == "recset":
                            for val in values:
                                if val["online"] == "Y":
                                    t_record = re.search(t_rec_pattern, val["spec"])
                                    if t_record.group(1) in answer:
                                        answer[t_record.group(1)].append(t_record.group(2))
                                    else:
                                        answer[t_record.group(1)] = [t_record.group(2)]

                    self._logger.info("Online Records JSON Downloaded - SharpNum:%s", sharpNum)
                    done = True
                else:
                    self._logger.critical("Online Records HTTP request Failed: %s - SharpNum:%s", resp.status_code,
                                          sharpNum)
                    done = False
                    try_count += 1

            except Exception as e:
                self._logger.critical("Online Records JSON Download Failed. Error:%s - SharpNum:%s", str(e),
                                      sharpNum)
                try_count += 1

        if done:
            return answer
        else:
            return None
        

    
    def download_records_for_range(self, start_time: datetime, end_time: datetime) -> List[int]:
        """
        Downloads harp data between the start and end time provided. If the data already exists on disk, then nothing
        is done.

        :param start_time: The start time to begin downloading data from

        :param end_time: The end time to end downloading data from

        :return: A list of harp numbers that had data downloaded
        """
        online_files = self.get_list_of_online_by_time_range(start_time, end_time)
        online_files = self.__filter_online(online_files)
        return self.__download_online_records(online_files)
    
    

    def download_records_for_sharpNum(self, sharpNum:str) -> List[int]:
        """
        Downloads harp data for sharp Number provided. If the data already exists on disk, then nothing
        is done.

        :param sharpNum: The Harp Num to downloading data for

        :return: A list of harp numbers that had data downloaded
        """
        online_files = self.get_list_of_online_by_sharpNum(sharpNum)
        online_files = self.__filter_online(online_files)
        return self.__download_online_records(online_files)