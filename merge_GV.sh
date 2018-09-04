#!/bin/bash
# This function is appending content from textfiles in OLDFILES to corresponding files in NEWFILES (files in both directories must have SAME name)
# Kamil Bonna, 04.09.2018

NEWFILES=data/graphNEW/
OLDFILES=data/graph/

NEWFILES=${NEWFILES}"*"
echo $NEWFILES
for newfile in $NEWFILES
	do
	echo "found: $newfile"
	oldfile=`basename $newfile`
	oldfile=$OLDFILES$oldfile
	echo "found corresponding: $oldfile"
	echo "[job] appending data..." 	
	cat $newfile >> $oldfile
	done

