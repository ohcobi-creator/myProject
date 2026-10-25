#!/system/bin/sh
# Ask the stock split screen service to turn split screen OFF (same call the stock toggle uses).
export PATH=/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin
O=/storage/usb0/SPLIT
mkdir $O
I=$O/off.txt
S=com.mobis.splitscreen.right.SplitScreenHome
echo "off start" > $I
service call $S 10 i32 0 i32 1 i32 0 >> $I 2>&1
echo "off done" >> $I
sync
sleep 3
sync
