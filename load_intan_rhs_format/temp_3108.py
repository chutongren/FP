# -*- coding: utf-8 -*-
import math
import struct
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, freqs, filtfilt
from scipy.signal import cheb2ord, cheby2
from scipy.fft import fft, fftfreq


# Intan python rhs read file: https://intantech.com/downloads.html?tabSelect=Software&yPos=123

# Modularity
# intanutil/report, header, data, filter module

# REPORT FUNCTION
def read_qstring(fid):
    """Reads Qt style QString.

    The first 32-bit unsigned number indicates the length of the string
    (in bytes). If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """
    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16):
        return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1):
        print(length)
        raise QStringError('Length too long.')

    # Convert length from bytes to 16-bit Unicode words.
    length = int(length / 2)

    data = []
    for _ in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    a = ''.join([chr(c) for c in data])

    return a


def print_record_time_summary(num_amp_samples, sample_rate, data_present):
    """Prints summary of how much recorded data is present in RHS file
    to console.
    """
    record_time = num_amp_samples / sample_rate

    # if data_present:
    #     print(f'File contains {record_time:0.3f} seconds of data.  '
    #         f'Amplifiers were sampled at {sample_rate / 1000:0.2f} kS/s.')
    # else:
    #     print(f'Header file contains no data.  '
    #         f'Amplifiers were sampled at {sample_rate / 1000:0.2f} kS/s.')

    if data_present:
        print('File contains {:0.3f} seconds of data.  '
              'Amplifiers were sampled at {:0.2f} kS/s.'
              .format(record_time, sample_rate / 1000))
    else:
        print('Header file contains no data.  '
              'Amplifiers were sampled at {:0.2f} kS/s.'
              .format(sample_rate / 1000))


def print_progress(i, target, print_step, percent_done):
    """Prints progress of an arbitrary process based on position i / target,
    printing a line showing completion percentage for each print_step / 100.
    """
    fraction_done = 100 * (1.0 * i / target)
    if fraction_done >= percent_done:
        print('{}% done...'.format(percent_done))
        percent_done += print_step

    return percent_done


class QStringError(Exception):
    """Exception returned when reading a QString fails because it is too long.
    """


# HEADER FUNCTIONS
def read_header(fid):
    """Reads the Intan File Format header from the given file.
    """
    check_magic_number(fid)

    header = {}

    read_version_number(header, fid)
    set_num_samples_per_data_block(header)

    read_sample_rate(header, fid)
    read_freq_settings(header, fid)

    read_notch_filter_frequency(header, fid)
    read_impedance_test_frequencies(header, fid)
    read_amp_settle_mode(header, fid)
    read_charge_recovery_mode(header, fid)

    create_frequency_parameters(header)

    read_stim_step_size(header, fid)
    read_recovery_current_limit(header, fid)
    read_recovery_target_voltage(header, fid)

    read_notes(header, fid)
    read_dc_amp_saved(header, fid)
    read_eval_board_mode(header, fid)
    read_reference_channel(header, fid)

    initialize_channels(header)
    read_signal_summary(header, fid)

    return header


def check_magic_number(fid):
    """Checks magic number at beginning of file to verify this is an Intan
    Technologies RHS data file.
    """
    magic_number, = struct.unpack('<I', fid.read(4))
    if magic_number != int('d69127ac', 16):
        raise UnrecognizedFileError('Unrecognized file type.')


def read_version_number(header, fid):
    """Reads version number (major and minor) from fid. Stores them into
    header['version']['major'] and header['version']['minor'].
    """
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    print('\nReading Intan Technologies RHS Data File, Version {}.{}\n'
          .format(version['major'], version['minor']))


def set_num_samples_per_data_block(header):
    """Determines how many samples are present per data block (always 128 for
    RHS files)
    """
    header['num_samples_per_data_block'] = 128


def read_sample_rate(header, fid):
    """Reads sample rate from fid. Stores it into header['sample_rate'].
    """
    header['sample_rate'], = struct.unpack('<f', fid.read(4))


def read_freq_settings(header, fid):
    """Reads amplifier frequency settings from fid. Stores them in 'header'
    dict.
    """
    (header['dsp_enabled'],
     header['actual_dsp_cutoff_frequency'],
     header['actual_lower_bandwidth'],
     header['actual_lower_settle_bandwidth'],
     header['actual_upper_bandwidth'],
     header['desired_dsp_cutoff_frequency'],
     header['desired_lower_bandwidth'],
     header['desired_lower_settle_bandwidth'],
     header['desired_upper_bandwidth']) = struct.unpack('<hffffffff',
                                                        fid.read(34))


def read_notch_filter_frequency(header, fid):
    """Reads notch filter mode from fid, and stores frequency (in Hz) in
    'header' dict.
    """
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60


def read_impedance_test_frequencies(header, fid):
    """Reads desired and actual impedance test frequencies from fid, and stores
    them (in Hz) in 'freq' dicts.
    """
    (header['desired_impedance_test_frequency'],
     header['actual_impedance_test_frequency']) = (
         struct.unpack('<ff', fid.read(8)))


def read_amp_settle_mode(header, fid):
    """Reads amp settle mode from fid, and stores it in 'header' dict.
    """
    header['amp_settle_mode'], = struct.unpack('<h', fid.read(2))


def read_charge_recovery_mode(header, fid):
    """Reads charge recovery mode from fid, and stores it in 'header' dict.
    """
    header['charge_recovery_mode'], = struct.unpack('<h', fid.read(2))


def create_frequency_parameters(header):
    """Copy various frequency-related parameters (set in other functions) to
    the dict at header['frequency_parameters'].
    """
    freq = {}
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']
    copy_from_header(header, freq, 'desired_dsp_cutoff_frequency')
    copy_from_header(header, freq, 'actual_dsp_cutoff_frequency')
    copy_from_header(header, freq, 'dsp_enabled')
    copy_from_header(header, freq, 'desired_lower_bandwidth')
    copy_from_header(header, freq, 'desired_lower_settle_bandwidth')
    copy_from_header(header, freq, 'actual_lower_bandwidth')
    copy_from_header(header, freq, 'actual_lower_settle_bandwidth')
    copy_from_header(header, freq, 'desired_upper_bandwidth')
    copy_from_header(header, freq, 'actual_upper_bandwidth')
    copy_from_header(header, freq, 'notch_filter_frequency')
    copy_from_header(header, freq, 'desired_impedance_test_frequency')
    copy_from_header(header, freq, 'actual_impedance_test_frequency')
    header['frequency_parameters'] = freq


def copy_from_header(header, freq_params, key):
    """Copy from header
    """
    freq_params[key] = header[key]


def read_stim_step_size(header, fid):
    """Reads stim step size from fid, and stores it in 'header' dict.
    """
    header['stim_step_size'], = struct.unpack('f', fid.read(4))


def read_recovery_current_limit(header, fid):
    """Reads charge recovery current limit from fid, and stores it in 'header'
    dict.
    """
    header['recovery_current_limit'], = struct.unpack('f', fid.read(4))


def read_recovery_target_voltage(header, fid):
    """Reads charge recovery target voltage from fid, and stores it in 'header'
    dict.
    """
    header['recovery_target_voltage'], = struct.unpack('f', fid.read(4))


def read_notes(header, fid):
    """Reads notes as QStrings from fid, and stores them as strings in
    header['notes'] dict.
    """
    header['notes'] = {'note1': read_qstring(fid),
                       'note2': read_qstring(fid),
                       'note3': read_qstring(fid)}


def read_dc_amp_saved(header, fid):
    """Reads whether DC amp data was saved from fid, and stores it in 'header'
    dict.
    """
    header['dc_amplifier_data_saved'], = struct.unpack('<h', fid.read(2))


def read_eval_board_mode(header, fid):
    """Stores eval board mode in header['eval_board_mode'].
    """
    header['eval_board_mode'], = struct.unpack('<h', fid.read(2))


def read_reference_channel(header, fid):
    """Reads name of reference channel as QString from fid, and stores it as
    a string in header['reference_channel'].
    """
    header['reference_channel'] = read_qstring(fid)


def initialize_channels(header):
    """Creates empty lists for each type of data channel and stores them in
    'header' dict.
    """
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['board_adc_channels'] = []
    header['board_dac_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []


def read_signal_summary(header, fid):
    """Reads signal summary from data file header and stores information for
    all signal groups and their channels in 'header' dict.
    """
    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    for signal_group in range(1, number_of_signal_groups + 1):
        add_signal_group_information(header, fid, signal_group)
    add_num_channels(header)
    print_header_summary(header)


def add_signal_group_information(header, fid, signal_group):
    """Adds information for a signal group and all its channels to 'header'
    dict.
    """
    signal_group_name = read_qstring(fid)
    signal_group_prefix = read_qstring(fid)
    (signal_group_enabled, signal_group_num_channels, _) = struct.unpack(
        '<hhh', fid.read(6))

    if signal_group_num_channels > 0 and signal_group_enabled > 0:
        for _ in range(0, signal_group_num_channels):
            add_channel_information(header, fid, signal_group_name,
                                    signal_group_prefix, signal_group)


def add_channel_information(header, fid, signal_group_name,
                            signal_group_prefix, signal_group):
    """Reads a new channel's information from fid and appends it to 'header'
    dict.
    """
    (new_channel, new_trigger_channel, channel_enabled,
     signal_type) = read_new_channel(
         fid, signal_group_name, signal_group_prefix, signal_group)
    append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type)


def read_new_channel(fid, signal_group_name, signal_group_prefix,
                     signal_group):
    """Reads a new channel's information from fid.
    """
    new_channel = {'port_name': signal_group_name,
                   'port_prefix': signal_group_prefix,
                   'port_number': signal_group}
    new_channel['native_channel_name'] = read_qstring(fid)
    new_channel['custom_channel_name'] = read_qstring(fid)
    (new_channel['native_order'],
     new_channel['custom_order'],
     signal_type, channel_enabled,
     new_channel['chip_channel'],
     _,  # ignore command_stream
     new_channel['board_stream']) = (
         struct.unpack('<hhhhhHh', fid.read(14)))
    new_trigger_channel = {}
    (new_trigger_channel['voltage_trigger_mode'],
     new_trigger_channel['voltage_threshold'],
     new_trigger_channel['digital_trigger_channel'],
     new_trigger_channel['digital_edge_polarity']) = (
         struct.unpack('<hhhh', fid.read(8)))
    (new_channel['electrode_impedance_magnitude'],
     new_channel['electrode_impedance_phase']) = (
         struct.unpack('<ff', fid.read(8)))

    return new_channel, new_trigger_channel, channel_enabled, signal_type


def append_new_channel(header, new_channel, new_trigger_channel,
                       channel_enabled, signal_type):
    """"Appends 'new_channel' to 'header' dict depending on if channel is
    enabled and the signal type.
    """
    if not channel_enabled:
        return

    if signal_type == 0:
        header['amplifier_channels'].append(new_channel)
        header['spike_triggers'].append(new_trigger_channel)
    elif signal_type == 1:
        raise UnknownChannelTypeError('No aux input signals in RHS format.')
    elif signal_type == 2:
        raise UnknownChannelTypeError('No Vdd signals in RHS format.')
    elif signal_type == 3:
        header['board_adc_channels'].append(new_channel)
    elif signal_type == 4:
        header['board_dac_channels'].append(new_channel)
    elif signal_type == 5:
        header['board_dig_in_channels'].append(new_channel)
    elif signal_type == 6:
        header['board_dig_out_channels'].append(new_channel)
    else:
        raise UnknownChannelTypeError('Unknown channel type.')


def add_num_channels(header):
    """Adds channel numbers for all signal types to 'header' dict.
    """
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dac_channels'] = len(header['board_dac_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(
        header['board_dig_out_channels'])


def header_to_result(header, result):
    """Merges header information from .rhs file into a common 'result' dict.
    If any fields have been allocated but aren't relevant (for example, no
    channels of this type exist), does not copy those entries into 'result'.
    """
    stim_parameters = {}
    stim_parameters['stim_step_size'] = header['stim_step_size']
    stim_parameters['charge_recovery_current_limit'] = \
        header['recovery_current_limit']
    stim_parameters['charge_recovery_target_voltage'] = \
        header['recovery_target_voltage']
    stim_parameters['amp_settle_mode'] = header['amp_settle_mode']
    stim_parameters['charge_recovery_mode'] = header['charge_recovery_mode']
    result['stim_parameters'] = stim_parameters

    result['notes'] = header['notes']

    if header['num_amplifier_channels'] > 0:
        result['spike_triggers'] = header['spike_triggers']
        result['amplifier_channels'] = header['amplifier_channels']

    result['notes'] = header['notes']
    result['frequency_parameters'] = header['frequency_parameters']
    result['reference_channel'] = header['reference_channel']

    if header['num_board_adc_channels'] > 0:
        result['board_adc_channels'] = header['board_adc_channels']

    if header['num_board_dac_channels'] > 0:
        result['board_dac_channels'] = header['board_dac_channels']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_channels'] = header['board_dig_in_channels']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_channels'] = header['board_dig_out_channels']

    return result


def print_header_summary(header):
    """Prints summary of contents of RHD header to console.
    """
    print('Found {} amplifier channel{}.'.format(
        header['num_amplifier_channels'],
        plural(header['num_amplifier_channels'])))
    if header['dc_amplifier_data_saved']:
        print('Found {} DC amplifier channel{}.'.format(
            header['num_amplifier_channels'],
            plural(header['num_amplifier_channels'])))
    print('Found {} board ADC channel{}.'.format(
        header['num_board_adc_channels'],
        plural(header['num_board_adc_channels'])))
    print('Found {} board DAC channel{}.'.format(
        header['num_board_dac_channels'],
        plural(header['num_board_dac_channels'])))
    print('Found {} board digital input channel{}.'.format(
        header['num_board_dig_in_channels'],
        plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(
        header['num_board_dig_out_channels'],
        plural(header['num_board_dig_out_channels'])))
    print('')


def plural(number_of_items):
    """Utility function to pluralize words based on the number of items.
    """
    if number_of_items == 1:
        return ''
    return 's'


class UnrecognizedFileError(Exception):
    """Exception returned when reading a file as an RHS header yields an
    invalid magic number (indicating this is not an RHS header file).
    """


class UnknownChannelTypeError(Exception):
    """Exception returned when a channel field in RHS header does not have
    a recognized signal_type value. Accepted values are:
    0: amplifier channel
    1: aux input channel (RHD only, invalid for RHS)
    2: supply voltage channel (RHD only, invalid for RHS)
    3: board adc channel
    4: board dac channel
    5: dig in channel
    6: dig out channel
    """


# DATA FUNCTION
def calculate_data_size(header, filename, fid):
    """Calculates how much data is present in this file. Returns:
    data_present: Bool, whether any data is present in file
    filesize: Int, size (in bytes) of file
    num_blocks: Int, number of 60 or 128-sample data blocks present
    num_samples: Int, number of samples present in file
    """
    bytes_per_block = get_bytes_per_data_block(header)

    # Determine filesize and if any data is present.
    filesize = os.path.getsize(filename)
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    # If the file size is somehow different than expected, raise an error.
    if bytes_remaining % bytes_per_block != 0:
        raise FileSizeError(
            'Something is wrong with file size : '
            'should have a whole number of data blocks')

    # Calculate how many data blocks are present.
    num_blocks = int(bytes_remaining / bytes_per_block)

    num_samples = calculate_num_samples(header, num_blocks)

    print_record_time_summary(num_samples,
                              header['sample_rate'],
                              data_present)

    return data_present, filesize, num_blocks, num_samples


def read_all_data_blocks(header, num_samples, num_blocks, fid):
    """Reads all data blocks present in file, allocating memory for and
    returning 'data' dict containing all data.
    """
    data, index = initialize_memory(header, num_samples)
    print("Reading data from file...")
    print_step = 10
    percent_done = print_step
    for i in range(num_blocks):
        read_one_data_block(data, header, index, fid)
        index = advance_index(index, header['num_samples_per_data_block'])
        percent_done = print_progress(i, num_blocks, print_step, percent_done)
    return data


def check_end_of_file(filesize, fid):
    """Checks that the end of the file was reached at the expected position.
    If not, raise FileSizeError.
    """
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining != 0:
        raise FileSizeError('Error: End of file not reached.')


def parse_data(header, data):
    """Parses raw data into user readable and interactable forms (for example,
    extracting raw digital data to separate channels and scaling data to units
    like microVolts, degrees Celsius, or seconds.)
    """
    print('Parsing data...')
    extract_digital_data(header, data)
    extract_stim_data(data)
    scale_analog_data(header, data)
    scale_timestamps(header, data)


def data_to_result(header, data, result):
    """Merges data from all present signals into a common 'result' dict. If
    any signal types have been allocated but aren't relevant (for example,
    no channels of this type exist), does not copy those entries into 'result'.
    """
    result['t'] = data['t']
    result['stim_data'] = data['stim_data']

    if header['dc_amplifier_data_saved']:
        result['dc_amplifier_data'] = data['dc_amplifier_data']

    if header['num_amplifier_channels'] > 0:
        result['compliance_limit_data'] = data['compliance_limit_data']
        result['charge_recovery_data'] = data['charge_recovery_data']
        result['amp_settle_data'] = data['amp_settle_data']
        result['amplifier_data'] = data['amplifier_data']

    if header['num_board_adc_channels'] > 0:
        result['board_adc_data'] = data['board_adc_data']

    if header['num_board_dac_channels'] > 0:
        result['board_dac_data'] = data['board_dac_data']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_data'] = data['board_dig_in_data']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_data'] = data['board_dig_out_data']

    return result


def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 128 sample datablock."""
    # RHS files always have 128 samples per data block.
    # Use this number along with numbers of channels to accrue a sum of how
    # many bytes each data block should contain.
    num_samples_per_data_block = 128

    # Timestamps (one channel always present): Start with 4 bytes per sample.
    bytes_per_block = bytes_per_signal_type(
        num_samples_per_data_block,
        1,
        4)

    # Amplifier data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_amplifier_channels'],
        2)

    # DC Amplifier data (absent if flag was off).
    if header['dc_amplifier_data_saved']:
        bytes_per_block += bytes_per_signal_type(
            num_samples_per_data_block,
            header['num_amplifier_channels'],
            2)

    # Stimulation data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_amplifier_channels'],
        2)

    # Analog inputs: Add 2 bytes per sample per enabled analog input channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_board_adc_channels'],
        2)

    # Analog outputs: Add 2 bytes per sample per enabled analog output channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_board_dac_channels'],
        2)

    # Digital inputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            num_samples_per_data_block,
            1,
            2)

    # Digital outputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            num_samples_per_data_block,
            1,
            2)

    return bytes_per_block


def bytes_per_signal_type(num_samples, num_channels, bytes_per_sample):
    """Calculates the number of bytes, per data block, for a signal type
    provided the number of samples (per data block), the number of enabled
    channels, and the size of each sample in bytes.
    """
    return num_samples * num_channels * bytes_per_sample


def read_one_data_block(data, header, index, fid):
    """Reads one 60 or 128 sample data block from fid into data,
    at the location indicated by index."""
    samples_per_block = header['num_samples_per_data_block']

    read_timestamps(fid,
                    data,
                    index,
                    samples_per_block)

    read_analog_signals(fid,
                        data,
                        index,
                        samples_per_block,
                        header)

    read_digital_signals(fid,
                         data,
                         index,
                         samples_per_block,
                         header)


def read_timestamps(fid, data, index, num_samples):
    """Reads timestamps from binary file as a NumPy array, indexing them
    into 'data'.
    """
    start = index
    end = start + num_samples
    format_sign = 'i'
    format_expression = '<' + format_sign * num_samples
    read_length = 4 * num_samples
    data['t'][start:end] = np.array(struct.unpack(
        format_expression, fid.read(read_length)))


def read_analog_signals(fid, data, index, samples_per_block, header):
    """Reads all analog signal types present in RHD files: amplifier_data,
    aux_input_data, supply_voltage_data, temp_sensor_data, and board_adc_data,
    into 'data' dict.
    """

    read_analog_signal_type(fid,
                            data['amplifier_data'],
                            index,
                            samples_per_block,
                            header['num_amplifier_channels'])

    if header['dc_amplifier_data_saved']:
        read_analog_signal_type(fid,
                                data['dc_amplifier_data'],
                                index,
                                samples_per_block,
                                header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['stim_data_raw'],
                            index,
                            samples_per_block,
                            header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['board_adc_data'],
                            index,
                            samples_per_block,
                            header['num_board_adc_channels'])

    read_analog_signal_type(fid,
                            data['board_dac_data'],
                            index,
                            samples_per_block,
                            header['num_board_dac_channels'])


def read_digital_signals(fid, data, index, samples_per_block, header):
    """Reads all digital signal types present in RHD files: board_dig_in_raw
    and board_dig_out_raw, into 'data' dict.
    """

    read_digital_signal_type(fid,
                             data['board_dig_in_raw'],
                             index,
                             samples_per_block,
                             header['num_board_dig_in_channels'])

    read_digital_signal_type(fid,
                             data['board_dig_out_raw'],
                             index,
                             samples_per_block,
                             header['num_board_dig_out_channels'])


def read_analog_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be an analog signal type within 'data', for example
    data['amplifier_data'] or data['aux_input_data']. Each sample is assumed
    to be of dtype 'uint16'.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    tmp = np.fromfile(fid, dtype='uint16', count=num_samples*num_channels)
    dest[range(num_channels), start:end] = (
        tmp.reshape(num_channels, num_samples))


def read_digital_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be a digital signal type within 'data', either
    data['board_dig_in_raw'] or data['board_dig_out_raw'].
    """

    if num_channels < 1:
        return
    end = start + num_samples
    dest[start:end] = np.array(struct.unpack(
        '<' + 'H' * num_samples, fid.read(2 * num_samples)))


def calculate_num_samples(header, num_data_blocks):
    """Calculates number of samples in file (per channel).
    """
    return int(header['num_samples_per_data_block'] * num_data_blocks)


def initialize_memory(header, num_samples):
    """Pre-allocates NumPy arrays for each signal type that will be filled
    during this read, and initializes index for data access.
    """
    print('\nAllocating memory for data...')
    data = {}

    # Create zero array for timestamps.
    data['t'] = np.zeros(num_samples, np.int_)

    # Create zero array for amplifier data.
    data['amplifier_data'] = np.zeros(
        [header['num_amplifier_channels'], num_samples], dtype=np.uint)

    # Create zero array for DC amplifier data.
    if header['dc_amplifier_data_saved']:
        data['dc_amplifier_data'] = np.zeros(
            [header['num_amplifier_channels'], num_samples], dtype=np.uint)

    # Create zero array for stim data.
    data['stim_data_raw'] = np.zeros(
        [header['num_amplifier_channels'], num_samples], dtype=np.int_)
    data['stim_data'] = np.zeros(
        [header['num_amplifier_channels'], num_samples], dtype=np.int_)

    # Create zero array for board ADC data.
    data['board_adc_data'] = np.zeros(
        [header['num_board_adc_channels'], num_samples], dtype=np.uint)

    # Create zero array for board DAC data.
    data['board_dac_data'] = np.zeros(
        [header['num_board_dac_channels'], num_samples], dtype=np.uint)

    # By default, this script interprets digital events (digital inputs
    # and outputs) as booleans. if unsigned int values are preferred
    # (0 for False, 1 for True), replace the 'dtype=np.bool_' argument
    # with 'dtype=np.uint' as shown.
    # The commented lines below illustrate this for digital input data;
    # the same can be done for digital out.

    # data['board_dig_in_data'] = np.zeros(
    #     [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
    #     dtype=np.uint)
    # Create 16-row zero array for digital in data, and 1-row zero array for
    # raw digital in data (each bit of 16-bit entry represents a different
    # digital input.)
    data['board_dig_in_data'] = np.zeros(
        [header['num_board_dig_in_channels'], num_samples],
        dtype=np.bool_)
    data['board_dig_in_raw'] = np.zeros(
        num_samples,
        dtype=np.uint)

    # Create 16-row zero array for digital out data, and 1-row zero array for
    # raw digital out data (each bit of 16-bit entry represents a different
    # digital output.)
    data['board_dig_out_data'] = np.zeros(
        [header['num_board_dig_out_channels'], num_samples],
        dtype=np.bool_)
    data['board_dig_out_raw'] = np.zeros(
        num_samples,
        dtype=np.uint)

    # Set index representing position of data (shared across all signal types
    # for RHS file) to 0
    index = 0

    return data, index


def scale_timestamps(header, data):
    """Verifies no timestamps are missing, and scales timestamps to seconds.
    """
    # Check for gaps in timestamps.
    num_gaps = np.sum(np.not_equal(
        data['t'][1:]-data['t'][:-1], 1))
    if num_gaps == 0:
        print('No missing timestamps in data.')
    else:
        print('Warning: {0} gaps in timestamp data found.  '
              'Time scale will not be uniform!'
              .format(num_gaps))

    # Scale time steps (units = seconds).
    data['t'] = data['t'] / header['sample_rate']


def scale_analog_data(header, data):
    """Scales all analog data signal types (amplifier data, stimulation data,
    DC amplifier data, board ADC data, and board DAC data) to suitable
    units (microVolts, Volts, microAmps).
    """
    # Scale amplifier data (units = microVolts).
    data['amplifier_data'] = np.multiply(
        0.195, (data['amplifier_data'].astype(np.int32) - 32768))
    data['stim_data'] = np.multiply(
        header['stim_step_size'],
        data['stim_data'] / 1.0e-6)

    # Scale DC amplifier data (units = Volts).
    if header['dc_amplifier_data_saved']:
        data['dc_amplifier_data'] = (
            np.multiply(-0.01923,
                        data['dc_amplifier_data'].astype(np.int32) - 512))

    # Scale board ADC data (units = Volts).
    data['board_adc_data'] = np.multiply(
        312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768))

    # Scale board DAC data (units = Volts).
    data['board_dac_data'] = np.multiply(
        312.5e-6, (data['board_dac_data'].astype(np.int32) - 32768))


def extract_digital_data(header, data):
    """Extracts digital data from raw (a single 16-bit vector where each bit
    represents a separate digital input channel) to a more user-friendly 16-row
    list where each row represents a separate digital input channel. Applies to
    digital input and digital output data.
    """
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_in_raw'],
                (1 << header['board_dig_in_channels'][i]['native_order'])
                ),
            0)

    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_out_raw'],
                (1 << header['board_dig_out_channels'][i]['native_order'])
                ),
            0)


def extract_stim_data(data):
    """Extracts stimulation data from stim_data_raw and stim_polarity vectors
    to individual lists representing compliance_limit_data,
    charge_recovery_data, amp_settle_data, stim_polarity, and stim_data
    """
    # Interpret 2^15 bit (compliance limit) as True or False.
    data['compliance_limit_data'] = np.bitwise_and(
        data['stim_data_raw'], 32768) >= 1

    # Interpret 2^14 bit (charge recovery) as True or False.
    data['charge_recovery_data'] = np.bitwise_and(
        data['stim_data_raw'], 16384) >= 1

    # Interpret 2^13 bit (amp settle) as True or False.
    data['amp_settle_data'] = np.bitwise_and(
        data['stim_data_raw'], 8192) >= 1

    # Interpret 2^8 bit (stim polarity) as +1 for 0_bit or -1 for 1_bit.
    data['stim_polarity'] = 1 - (2 * (np.bitwise_and(
        data['stim_data_raw'], 256) >> 8))

    # Get least-significant 8 bits corresponding to the current amplitude.
    curr_amp = np.bitwise_and(data['stim_data_raw'], 255)

    # Multiply current amplitude by the correct sign.
    data['stim_data'] = curr_amp * data['stim_polarity']


def advance_index(index, samples_per_block):
    """Advances index used for data access by suitable values per data block.
    """
    # For RHS, all signals sampled at the same sample rate:
    # Index should be incremented by samples_per_block every data block.
    index += samples_per_block
    return index


class FileSizeError(Exception):
    """Exception returned when file reading fails due to the file size
    being invalid or the calculated file size differing from the actual
    file size.
    """


# FILTER FUNCTION
def apply_notch_filter(header, data):
    """Checks header to determine if notch filter should be applied, and if so,
    apply notch filter to all signals in data['amplifier_data'].
    """
    # If data was not recorded with notch filter turned on, return without
    # applying notch filter. Similarly, if data was recorded from Intan RHX
    # software version 3.0 or later, any active notch filter was already
    # applied to the saved data, so it should not be re-applied.
    if (header['notch_filter_frequency'] == 0
            or header['version']['major'] >= 3):
        return

    # Apply notch filter individually to each channel in order
    print('Applying notch filter...')
    print_step = 10
    percent_done = print_step
    for i in range(header['num_amplifier_channels']):
        data['amplifier_data'][i, :] = notch_filter(
            data['amplifier_data'][i, :],
            header['sample_rate'],
            header['notch_filter_frequency'],
            10)

        percent_done = print_progress(i, header['num_amplifier_channels'],
                                      print_step, percent_done)


def notch_filter(signal_in, f_sample, f_notch, bandwidth):
    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector 'signal_in'.

    f_sample = sample rate of data (input Hz or Samples/sec)
    f_notch = filter notch frequency (input Hz)
    bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(signal_in, 30000, 60, 10);
    """
    # Calculate parameters used to implement IIR filter
    t_step = 1.0/f_sample
    f_c = f_notch*t_step
    signal_length = len(signal_in)
    iir_parameters = calculate_iir_parameters(bandwidth, t_step, f_c)

    # Create empty signal_out NumPy array
    signal_out = np.zeros(signal_length)

    # Set the first 2 samples of signal_out to signal_in.
    # If filtering a continuous data stream, change signal_out[0:1] to the
    # previous final two values of signal_out
    signal_out[0] = signal_in[0]
    signal_out[1] = signal_in[1]

    # Run filter.
    for i in range(2, signal_length):
        signal_out[i] = calculate_iir(i, signal_in, signal_out, iir_parameters)

    return signal_out


def calculate_iir_parameters(bandwidth, t_step, f_c):
    """Calculates parameters d, b, a0, a1, a2, a, b0, b1, and b2 used for
    IIR filter and return them in a dict.
    """
    parameters = {}
    d = math.exp(-2.0*math.pi*(bandwidth/2.0)*t_step)
    b = (1.0 + d*d) * math.cos(2.0*math.pi*f_c)
    a0 = 1.0
    a1 = -b
    a2 = d*d
    a = (1.0 + d*d)/2.0
    b0 = 1.0
    b1 = -2.0 * math.cos(2.0*math.pi*f_c)
    b2 = 1.0

    parameters['d'] = d
    parameters['b'] = b
    parameters['a0'] = a0
    parameters['a1'] = a1
    parameters['a2'] = a2
    parameters['a'] = a
    parameters['b0'] = b0
    parameters['b1'] = b1
    parameters['b2'] = b2
    return parameters


def calculate_iir(i, signal_in, signal_out, iir_parameters):
    """Calculates a single sample of IIR filter passing signal_in through
    iir_parameters, resulting in signal_out.
    """
    sample = ((
        iir_parameters['a'] * iir_parameters['b2'] * signal_in[i - 2]
        + iir_parameters['a'] * iir_parameters['b1'] * signal_in[i - 1]
        + iir_parameters['a'] * iir_parameters['b0'] * signal_in[i]
        - iir_parameters['a2'] * signal_out[i - 2]
        - iir_parameters['a1'] * signal_out[i - 1])
        / iir_parameters['a0'])

    return sample


# FOLLOWING FUNCTIONS WRITTEN BY OURSELVES
# FILTER(Butter)  （1）1000-3000（2）300-5000


import numpy as np
from scipy.signal import iirnotch, filtfilt, butter

def butter_bandpass_filter(data, num_channels, fs=30000):
    """
    Apply a Butterworth bandpass filter to retain frequencies between 30Hz and 8000Hz.

    Parameters:
    - data: ndarray, shape (num_channels, n_samples)
        Multi-channel signal data.
    - num_channels: int
        Number of channels in the data.
    - fs: int, optional, default=30000
        Sampling frequency in Hz.

    Returns:
    - filtered_data: ndarray, shape (num_channels, n_samples)
        Bandpass filtered multi-channel signal data.
    """
    lowcut = 100
    highcut = 8000
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Initialize the array to hold the filtered data
    filtered_data = np.zeros_like(data)

    # Generate the filter coefficients only once
    b, a = butter(N=2, Wn=[low, high], btype='band')

    # Apply the filter to each channel
    for i in range(num_channels):
        data_single_channel = data[i]
        filtered_data[i] = filtfilt(b, a, data_single_channel)

    return filtered_data

def notch_filter_revised(data, num_channels, fs=30000):
    """
    Apply notch filters to remove 50Hz and its harmonics below 2000Hz from multi-channel data.

    Parameters:
    - data: ndarray, shape (num_channels, n_samples)
        Multi-channel signal data.
    - num_channels: int
        Number of channels in the data.
    - fs: int, optional, default=30000
        Sampling frequency in Hz.

    Returns:
    - filtered_data: ndarray, shape (num_channels, n_samples)
        Filtered multi-channel signal data.
    """
    harmonics = [f for f in range(50, 2001, 50)]  # Frequencies to notch out
    filtered_data = np.copy(data)

    for harmonic in harmonics:
        # Design notch filter for the given harmonic
        f0 = harmonic  # Target frequency to remove
        Q = 30  # Quality factor, determines the bandwidth of the notch
        b, a = iirnotch(w0=f0 / (fs / 2), Q=Q)

        # Apply the notch filter to each channel
        for i in range(num_channels):
            data_single_channel = filtered_data[i]  # Single-channel data
            filtered_data[i] = filtfilt(b, a, data_single_channel)

    return filtered_data



# FILTER(CHEBY2) 
def cheby2_lowpass_filter(data, fs, wp, ws, gpass=3, gstop=40, rs=60, btype='lowpass'):
    wp = wp / (fs / 2)
    ws = ws / (fs / 2)
    N, Wn = cheb2ord(wp, ws, gpass, gstop) 
    print(N)
    print(Wn)
    b, a = cheby2(N, rs, Wn, btype) 
    data_preprocessed = filtfilt(b, a, data)  
    return data_preprocessed

def cheby2_highpass_filter(data, fs, wp, ws, gpass=3, gstop=40, rs=60, btype='highpass'):
    wp = wp / (fs / 2)
    ws = ws / (fs / 2)
    N, Wn = cheb2ord(wp, ws, gpass, gstop) 
    b, a = cheby2(N, rs, Wn, btype) 
    data_preprocessed = filtfilt(b, a, data) 
    return data_preprocessed


# 1. General Figure Plotting
# P: y轴刻度标签有点打架，fontsize 不起作用
def plot_filtered_data(filtered_data, num_channels, dataset_num):
    # Set the colormap for different channels
    # choose color node: jet: too colorful, cividis: dark blur -> yellow, viridis: purple->green->yellow, plasma: pink
    # colors = plt.cm.jet(np.linspace(0, 1, num_channels))
    fig, ax = plt.subplots(num_channels, 1, figsize=(15, 2*4), sharex=True)

    for i in range(num_channels):
        data = filtered_data[i, :]
        # data = a['amplifier_data'][i, :]
        # print(f"Signal range before filtering: {np.min(data)} to {np.max(data)}")

        # data = butter_bandpass_filter(data, fs, lowcut=lowcut, highcut=highcut)
        # amplification_factor = 1
        # filtered_data_amplified = filtered_data * amplification_factor
        # Calculate global min and max across all channels
        global_min = np.min(filtered_data)
        global_max = np.max(filtered_data)
        
        # Plot with thinner lines and different colors
        ax[i].plot(a['t'], data, linewidth=0.5) # , color=colors[i]
        ax[i].set_ylim(global_min, global_max) 
        # ax[i].set_ylabel('CH{}'.format(i+1), rotation=0, labelpad=20, va='center')
        # ax[i].yaxis.set_label_position('right')
        # ax[i].tick_params(axis='y', labelsize=8)  # 调整y轴刻度标签的字体大小

        # Hide the y-axis ticks and labels
        ax[i].set_yticks([])  # Remove y-axis tick labels

        # Add custom channel label on the y-axis (replace the tick labels) labelpad是ylabel的位置，va=“center”是纵向居中
        ax[i].set_ylabel('CH{}'.format(i+1), rotation=0, labelpad=20, va='center') # , labelpad=30, va='center', fontsize=10

    fig.text(0.07, 0.5, 'Voltage (μV)', va='center', rotation='vertical')
    
    ax[-1].set_xlabel('Time (s)')
    plt.subplots_adjust(hspace=0) # (hspace=0.8) 增加子图之间的垂直间距
    plt.savefig('../output_figs/task1_1000-3000Hzfiltered_0709/fig_general_{}.png'.format(dataset_num), dpi=300, bbox_inches='tight')
    plt.show()
    
    
# 2. 20s slice Plotting
def plot_20s(filtered_data, channels_selected, fs, dataset_num):
    start_time = 159-120 # data recorded start from 120, not from 0
    end_time = 179-120   # 159,179
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    # print(f"Expected t length: {end_idx - start_idx}")
    # print(f"Actual data length: {len(filtered_data[start_idx:end_idx])}")
    # Calculate global min and max for y-axis limits across selected channels
    global_min = np.min(filtered_data)
    global_max = np.max(filtered_data)
    
    channels_selected = channels_selected # better performance [0,1,4,5,6,7,8]
    for i in channels_selected:
        data = filtered_data[i, :]
        # filtered_data = butter_bandpass_filter(data, fs, lowcut=lowcut, highcut=highcut)
        t = np.linspace(0, 20, end_idx - start_idx) 
        plt.figure(figsize=(12, 7))
        plt.plot(t, data[start_idx:end_idx], linewidth=0.5) 
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (μV)')
        plt.xticks(range(0, 21, 2))  
        plt.ylim(-200, 200) 
        # plt.ylim(-150, 150)
        # plt.title('An arbitrary 20 seconds plot in data{} channel {}'.format(i))
        plt.savefig('../output_figs/task1_300-5000Hzfiltered_1409/fig_20s_data{}_channel{}.png'.format(dataset_num, i+1), dpi=300, bbox_inches='tight')
        plt.show()
    return data[start_idx:end_idx]


# 3. ERP Plotting
def find_peaks_in_data(data, peak_height):
    """
    Find peaks in the data for a specified channel

    Parameters:
    data: np.array, data for a single channel
    peak_height: float, threshold for peak detection

    Returns:
    peaks: list, indices of detected peaks
    """
    
    inverted_data = -data # for finding the lowest point, not the highest point 
    peaks, _ = find_peaks(inverted_data, height=peak_height)
    
    return peaks

def plot_single_peak(filtered_data, peak_index, fs, channel_num):
    pre_peak_samples = int(0.001 * fs)  # Number of samples per 1ms on the left  60
    post_peak_samples = int(0.001 * fs) # Number of samples per 1ms on the right
    # why 1ms? the AP process, 2ms is too much
    
    if peak_index - pre_peak_samples >= 0 and peak_index + post_peak_samples < len(filtered_data):
        spike_segment = filtered_data[peak_index - pre_peak_samples : peak_index + post_peak_samples]
        
        time_axis = np.linspace(0, pre_peak_samples + post_peak_samples, pre_peak_samples + post_peak_samples) / fs * 1000

        plt.figure()
        plt.plot(time_axis, spike_segment, color='blue')
        plt.title('ERP for Peak{} of Channel{}'.format(peak_index, channel_num[0]))
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (μV)')
        plt.grid(True)
        # plt.savefig('figs/plot_single_peak{}_channel{}.png'.format(peak_index,channel_num[0]), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f'Peak index {peak_index} is out of range for the given data.')


def plot_peaks_selected(filtered_data, fs, spikes_selected, channel_num, dataset_num, mode, low_amp_height_1=None, low_amp_height_2=None):
    if mode == 3: # caculate again
        if low_amp_height_1 is None or low_amp_height_2 is None:
            raise ValueError("For mode 3, both low_amp_height_1 and low_amp_height_2 must be provided.")

        # 查找低幅值spikes
        peaks1 = find_peaks_in_data(filtered_data, peak_height=low_amp_height_1)  # 查找大于low_amp_height_1的peaks
        peaks2 = find_peaks_in_data(filtered_data, peak_height=low_amp_height_2)  # 查找大于low_amp_height_2的peaks
        peaks_lowamplitude = np.setdiff1d(peaks1, peaks2)  # 从peaks1中删除peaks2中的峰值
        spikes_selected = peaks_lowamplitude # 無所謂，对于mode3情况，自己找出spikes_selected，覆盖了传入进来的参数
        print(f'Low amplitude peaks: {len(spikes_selected)}')  # 打印低幅值spikes的数量


    pre_peak_samples = int(0.001 * fs) 
    post_peak_samples = int(0.001 * fs) 
    
    num_peaks = len(spikes_selected)
    colors = plt.cm.jet(np.linspace(0, 1, num_peaks))  # 生成与峰值数量相同的颜色
    
    
    plt.figure(figsize=(8, 5))
    
    for i, peak_index in enumerate(spikes_selected):
        if peak_index - pre_peak_samples >= 0 and peak_index + post_peak_samples < len(filtered_data):
            spike_segment = filtered_data[peak_index - pre_peak_samples : peak_index + post_peak_samples]
            
            time_axis = np.linspace(0, pre_peak_samples + post_peak_samples, pre_peak_samples + post_peak_samples) / fs * 1000
            
            plt.plot(time_axis, spike_segment, color=colors[i], label=f'Peak {i+1}')
        else:
            print(f'Peak index {peak_index} is out of range for the given data.')

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (μV)')
    plt.xticks(np.arange(0, 2.2, 0.2)) 
    
    # if mode == 1:
    #     plt.savefig('../output_figs/task1_300-5000Hzfiltered_1409/erp_APtrend_data{}_channel{}.png'.format(dataset_num, channel_num[0]+1), dpi=300, bbox_inches='tight')
    # elif mode == 2:
    #     plt.savefig('../output_figs/task1_1000-3000Hzfiltered_0709/erp_highamplitude_data{}_channel{}.png'.format(dataset_num, channel_num[0]+1), dpi=300, bbox_inches='tight')
    # elif mode == 3:
    #     if low_amp_height_1 is None or low_amp_height_2 is None:
    #         raise ValueError("For mode 3, both low_amp_height_1 and low_amp_height_2 must be provided.")
    #     # plt.savefig('../output_figs/task1_1000-3000Hzfiltered_0709/erp_lowamplitude_data{}_channel{}.png'.format(dataset_num, channel_num[0]+1), dpi=300, bbox_inches='tight')
    
    # plt.legend(loc='upper right')
    plt.show()


def delete_spikes(spikes_selected, spikes_selected_deleted_index):
    return np.delete(spikes_selected, spikes_selected_deleted_index)


def plot_fft_combined(data, fs):

    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq

    combined_data = np.mean(data, axis=0)

    L = len(combined_data)
    
    sig_fft = fft(combined_data)
    
    freqs = fftfreq(L, 1/fs)
    

    sig_fft = 2 * np.abs(sig_fft[:L // 2]) / L
    freqs = freqs[:L // 2]
    
    plt.figure()
    plt.plot(freqs, sig_fft, color='blue')
    plt.title('Combined FFT Spectrum of All Channels')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    output_path = '../output_figs/task1_300-5000Hzfiltered_1409/fft.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"FFT plot saved to {output_path}")
    plt.show()

# load_intan_rhs_format.py: read_data + main  
def read_data(filename):
    """Reads Intan Technologies RHS2000 data file generated by acquisition
    software (IntanRHX, or legacy Stimulation/Recording Controller software).

    Data are returned in a dictionary, for future extensibility.
    """
    # Start measuring how long this read takes.
    tic = time.time()

    # Open file for reading.
    with open(filename, 'rb') as fid:

        # Read header and summarize its contents to console.
        header = read_header(fid)

        # Calculate how much data is present and summarize to console.
        data_present, filesize, num_blocks, num_samples = (
            calculate_data_size(header, filename, fid))

        # If .rhs file contains data, read all present data blocks into 'data'
        # dict, and verify the amount of data read.
        print('FINISHED HEADER')
        if data_present:
            data = read_all_data_blocks(header, num_samples, num_blocks, fid)
            check_end_of_file(filesize, fid)

    # Save information in 'header' to 'result' dict.
    result = {}
    header_to_result(header, result)

    # If .rhs file contains data, parse data into readable forms and, if
    # necessary, apply the same notch filter that was active during recording.
    if data_present:
        parse_data(header, data)
        apply_notch_filter(header, data)

        # Save recorded data in 'data' to 'result' dict.
        data_to_result(header, data, result)

    # Otherwise (.rhs file is just a header for One File Per Signal Type or
    # One File Per Channel data formats, in which actual data is saved in
    # separate .dat files), just return data as an empty list.
    else:
        data = []

    # Report how long read took.
    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))

    # Return 'result' dict.
    return result


if __name__ == '__main__':
    # a: original data; filtered_data: processed after butter filter;
    a = read_data(sys.argv[1])
    print(a.keys())
    # ['stim_data', 'amp_settle_data', 'amplifier_data', 'spike_triggers', 
    # 'notes', 'frequency_parameters', 'stim_parameters', 'charge_recovery_data',
    # 'amplifier_channels', 'compliance_limit_data', 'reference_channel', 't'] 
    
    # 0. filter 
    filter_mode = 2 # = 1:1000-3000; = 2: 300-5000  
    dataset_num = 3 # four datasets in total
    fs = 30000

    # plot_fft_combined(a['amplifier_data'][10,:], fs)
    # plot_fft_combined(a['amplifier_data'], fs)
    num_channels = a['amplifier_data'].shape[0] # 14(also 2 reference channels)
    # Example usage (assuming 'a["amplifier_data"]' contains your data):
    # Apply bandpass filter first
    bandpassed_data = butter_bandpass_filter(a['amplifier_data'], num_channels=num_channels)
    # Then apply notch filter
    filtered_data = notch_filter_revised(bandpassed_data, num_channels=num_channels)

    # filtered_data = butter_bandpass_filter(a['amplifier_data'], num_channels=num_channels, mode = filter_mode)  
    # Example usage (assuming 'a["amplifier_data"]' contains your data):
    # filtered_data = notch_filter_revised(a['amplifier_data'], num_channels=num_channels)

    # filtered_data = remove_50hz_harmonics(a['amplifier_data'], fs=30000, max_freq=2000, quality_factor=30)
    # plot_fft_combined(filtered_data, fs)
    
    # 1. general figure
    # plot_filtered_data(a['amplifier_data'], num_channels=num_channels, dataset_num=dataset_num) 
    plot_filtered_data(filtered_data, num_channels=num_channels, dataset_num=dataset_num)
    
    # 2. 20s figure
    channel_selected = [7] # [0,1,4,5,6,7,8]  manual input 
    data_selected20 = plot_20s(filtered_data, channel_selected, fs, dataset_num=dataset_num)
    # good performance: 4,8
    
    # Step two must be run before go through step three
    # 3. ERP figure
    erp_mode = 1 # 1: APtrend; 2: highamplitude; 3:lowamplitude, set two amp_height
    # 138, 95 selected
    peaks = find_peaks_in_data(data_selected20, peak_height=50) # ‘peak_height’ requires manual input, amp(spikes) > peak_height will be selected
    # # print(peaks) # print index of peaks 
    # print(len(peaks))
    
    # # plot_single_peak(data_selected20, peak_index=64878, fs=30000, channel_num=channel_selected) # ‘peak_index’ requires manual input
    spikes_selected = peaks
    # spikes_selected_deleted_index = [7,12,13]
    # spikes_selected = delete_spikes(peaks, spikes_selected_deleted_index)
    # # spikes_selected = [26994, 33376, 70878, 90690, 129801, 298983, 405590, 436397, 448779, 481932, 492909, 499529, 589276, 594441] 
    # plot_peaks_selected(data_selected20, fs, spikes_selected=spikes_selected, channel_num=channel_selected, dataset_num=dataset_num, mode = erp_mode) # mode = 3
    # 94,96     6
    # 
    # plot_peaks_selected(data_selected20, fs, spikes_selected=spikes_selected, channel_num=channel_selected, dataset_num=dataset_num, mode = erp_mode) # mode = 1/2
    
    # 1000-3000: func 1: APtrend;
    # 300-5000: func 2: highamplitude; 3:lowamplitude 两个参数   save的路径
    
    # Step 1: Find peaks with height 80 and 120
    # peaks1 = find_peaks_in_data(data_selected20, peak_height=50)   # Find peaks greater than or equal to 80
    # peaks2 = find_peaks_in_data(data_selected20, peak_height=51)  # Find peaks greater than or equal to 120

    # # Step 2: Remove common peaks from peaks1 (i.e., keep only spikes with heights between 80 and 120)
    # peaks_lowamplitude = np.setdiff1d(peaks1, peaks2)  # Keep spikes in peaks1 that are not in peaks2
    # print(len(peaks_lowamplitude))
    
    # Step 3: Plot filtered peaks
    # spikes_selected2 = peaks_lowamplitude
    # spikes_selected_deleted_index = [9,14]
    # spikes_selected2 = delete_spikes(peaks_lowamplitude, spikes_selected_deleted_index)
    # plot_peaks_selected(data_selected20, fs, spikes_selected=spikes_selected2, channel_num=channel_selected, dataset_num=dataset_num, mode = 3)

