#!/bin/bash

​

set -xu

​

allCommand=

while [ "$#" -gt 0 ];

do

    allCommand=" $allCommand $1 "

    shift

done

​

$allCommand


