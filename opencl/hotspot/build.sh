#!/usr/bin/env bash

BOARD=de5a_net_ddr4
BG=""
EMAIL=""
VERSIONS="0 1 2 3 5 7 9 11"
BSIZE=""
SIMD=""
CUSIZE=""
PROFILE=0

ARGS=$(getopt -o hb:e:v:s: -l "help,board:,bg,email:,version:,simd:,bsize:,ssize:,cusize:,profile"  -- "$@");

function usage() {
	echo "Usage"
	echo -e "\tbuild.sh [options]"
	echo ""
	echo "OPTIONS"
	echo -e "\th, --help"
	echo -e "\t\tDisplay this help message."
	echo -e "\tb, --board <board>"
	echo -e "\t\tSet the FPGA board name such as de5a_net_ddr4 (default: de5a_net_ddr4)."
	echo -e "\t--bg"
	echo -e "\t\tRun each make in backgournd."
	echo -e "\t-e, --email <address>"
	echo -e "\t\tSend notificaiton emails to the address."
	echo -e "\t-v, --version <versions>"
	echo -e "\t\tSpecify the versions to build. E.g., -v \"4 7 9\"."
	echo -e "\t-s, --simd <simd-lanes>"
	echo -e "\t\tSpecify the SIMD parameters."
	echo -e "\t--bsize <block-sizes>"
	echo -e "\t\tSSpecify the BSIZE parameters."
	echo -e "\t--ssize <stage-sizes>"
	echo -e "\t\tSSpecify the SSIZE parameters."
	echo -e "\t--cusize <compute-unit-sizes>"
	echo -e "\t\tSSpecify the CUSIZE parameters."
	echo -e "\t--profile"
	echo -e "\t\tEnable profiling."
}

#Bad arguments
if [ $? -ne 0 ];
then
	usage
	exit 1
fi

eval set -- "$ARGS";

while true; do
	case "$1" in
		-b|--board)
			BOARD=$2
			shift 2
			;;
		--bg)
			echo "Run background"
			BG="yes"
			shift
			;;
		-e|--email)
			EMAIL=$2
			echo Sending email to $EMAIL
			shift 2
			;;
		-v|--versions)
			VERSIONS=$2
			echo "Versions: $VERSIONS"
			shift 2
			;;
		--bsize)
			BSIZE=$2
			echo "BSIZE: $BSIZE"
			shift 2
			;;
		-s|--simd)
			SIMD=$2
			echo "SIMD lenths: $SIMD"
			shift 2
			;;
		--ssize)
			SSIZE=$2
			echo "SSIZE: $SSIZE"
			shift 2
			;;
		--cusize)
			CUSIZE=$2
			echo "CUSIZE: $CUSIZE"
			shift 2
			;;
		--profile)
			PROFILE=1
			shift
			;;
		-h|--help)
			usage
			exit
			;;
		--)
			shift
			break
			;;
	esac
done

export ALTERA=1
export BOARD

function run_make_email() {
	make_email_com=../../common/make_email.sh
	if [ -n "$BG" ]; then
		$make_email_com $EMAIL $* &
	else
		$make_email_com $EMAIL $*
	fi
}

function run_make() {
	echo run_make $*
	if [ -n "$EMAIL" ]; then
		run_make_email $*
	else
		OUT=$1
		shift
		if [ -n "$BG" ]; then
			make $* > $OUT 2>&1 &
		else
			make $* > $OUT 2>&1
		fi
	fi
}

function get_bsize() {
	local v=$1
	local bsize=$2
	if [ -n "$bsize" ]; then
		echo $bsize
		return
	fi
	case $v in
		0|2|3|5)
			bsize=16
			;;
		7|9|11)
			bsize=512
			;;
		*)
			bsize=""
	esac
	echo $bsize
	return
}

function get_simd() {
	local v=$1
	local simd=$2
	if [ -n "$simd" ]; then
		echo $simd
		return
	fi
	case $v in
		2)
			simd=4
			;;
		*)
			simd=""
	esac
	echo $simd
	return
}

function get_ssize() {
	local v=$1
	local ssize=$2
	if [ -n "$ssize" ]; then
		echo $ssize
		return
	fi
	case $v in
		7|9|11)
			ssize=8
			;;
		*)
			ssize=""
	esac
	echo $ssize
	return
}

function get_log_file_name() {
	local name="build"
	until [ -z "$1" ]; do
	   name="${name}_$1"
	   shift
	done
	if [ $PROFILE -eq 1 ]; then
	   name="${name}_PROFILE"
	fi
	name="${name}.out"
	echo $name
}

function get_cusize() {
	local v=$1
	local cusize=$2
	if [ -n "$cusize" ]; then
		echo $cusize
		return
	fi
	case $v in
		2)
			cusize=1
			;;
		*)
			cusize=""
	esac
	echo $cusize
	return
}

function build() {
	for v in $VERSIONS; do
		local bsize=$(get_bsize $v "$BSIZE")
		local simd=$(get_simd $v "$SIMD")
		local ssize=$(get_ssize $v "$SSIZE")
		local cusize=$(get_cusize $v "$CUSIZE")
		case $v in
			1)
				OUT=$(get_log_file_name v=${v})
				run_make $OUT v=$v PROFILE=$PROFILE kernel
				;;
			2)
				for bs in $bsize; do
					for s in $simd; do
						for cu in $cusize; do
							OUT=$(get_log_file_name v$v BSIZE$bs SIMD$s CUSIZE$cu)
							run_make $OUT v=$v BSIZE=$bs SIMD=$s PROFILE=$PROFILE CUSIZE=$cu kernel
						done
					done
				done
				;;
			0|3|5)
				for bs in $bsize; do
					OUT=$(get_log_file_name v$v BSIZE$bs)
					run_make $OUT v=$v BSIZE=$bs PROFILE=$PROFILE kernel
				done
				;;
			7|9|11)
				for bs in $bsize; do
					for s in $ssize; do
						OUT=$(get_log_file_name v$v BSIZE$bs SSIZE$s)
						run_make $OUT v=$v BSIZE=$bs SSIZE=$s PROFILE=$PROFILE kernel
					done
				done
				;;
			*)
				echo "Error: unsupported version number"
				exit -1
		esac
	done
}

build
