#!/system/bin/sh
# Read-only: ask the split screen service for its current state. Changes nothing.
export PATH=/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin
O=/storage/usb0/SPLIT
mkdir $O
I=$O/status.txt
S=com.mobis.splitscreen.right.SplitScreenHome
echo "status start" > $I
echo "== service list" >> $I
service list >> $I 2>&1
echo "== isSplitScreenOn(0)" >> $I
service call $S 6 i32 0 >> $I 2>&1
echo "== isSplitScreenOn(1)" >> $I
service call $S 6 i32 1 >> $I 2>&1
echo "== isSSUserOn(0,0)" >> $I
service call $S 7 i32 0 i32 0 >> $I 2>&1
echo "== isSplitButtonOn()" >> $I
service call $S 8 >> $I 2>&1
getprop ro.device.splitscreen >> $I
echo "status done" >> $I
sync
sleep 3
sync
