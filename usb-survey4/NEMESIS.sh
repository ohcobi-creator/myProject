#!/system/bin/sh
# Read-only copy of home-screen files. Writes only to the USB drive.
export PATH=/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin
O=/storage/usb0/HOME
mkdir $O
I=$O/info.txt
echo "copy start" > $I
for f in /system/app/Launcher_WP3.apk /system/app/Launcher_WP3.odex \
         /system/priv-app/SystemUI.apk /system/priv-app/SystemUI.odex; do
  cp $f $O/ >> $I 2>&1
  echo "$f -> $?" >> $I
done
ls -l $O >> $I 2>&1
echo "copy done" >> $I
sync
sleep 5
sync
