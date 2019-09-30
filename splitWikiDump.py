#!/usr/bin/env python3

import argparse
import math
import os
import sys


def parse_args():
    """ Parse program arguments """

    parser = argparse.ArgumentParser(
            description='split file to a given number of parts ')
    parser.add_argument('input_fnames', nargs=1,
            help='input file')
    parser.add_argument('--hosts', help='list of hosts',
            default='hosts.txt')
    parser.add_argument('-l', '--lastheadertag', help='tag that ends the header',
            default='</siteinfo>\n')
    parser.add_argument('-e', '--endtag', help='tag that should not be crossed',
            default='</page>\n')
    parser.add_argument('-t', '--tailtag', help='the tail tag',
            default='</mediawiki>\n')

    return parser.parse_args()


def gen_partfiles(input_fname, hosts):
    """ Generate and open files for parts """
    part_numb = 0
    for host in hosts:
        part_fname = input_fname + '.' + host
        yield open(part_fname, 'wb')


# parse program arguments
args = parse_args()
input_fname = args.input_fnames[0]

hosts=[]
with open(args.hosts, 'r') as hostsfile:
    for host in hostsfile:
        host = host.rstrip()
        hosts.append(host)

dumpsize = os.path.getsize(input_fname)

buffsize = 4096
buffreads = int(math.ceil(dumpsize / len(hosts) / buffsize))

partfiles = gen_partfiles(input_fname, hosts)
outfile = next(partfiles) 

endtag = args.endtag.encode('utf-8')
lastheadertag = args.lastheadertag.encode('utf-8')
tailtag = args.tailtag.encode('utf-8')

overlap = len(endtag) - 1
buff = None

fill_header = True
header = ''

with open(input_fname, 'rb') as f:
    for p in range(len(hosts)):
        for r in range(buffreads):
            buff = f.read(buffsize)
            if buff:
                outfile.write(buff)
                if fill_header:
                    pos = buff.find(lastheadertag)
                    if pos >= 0:
                        header = buff[0:pos+len(lastheadertag)]
                    fill_header = False
        buff = f.read(buffsize)
        while buff:
            pos = buff.find(endtag)
            if pos >= 0:
                outfile.write(buff[0:pos+len(endtag)])
                outfile.write(tailtag)
                outfile = next(partfiles) 
                outfile.write(header)
                outfile.write(buff[pos+len(endtag):])
                break
            else:
                outfile.write(buff)
            buff = f.read(buffsize)

