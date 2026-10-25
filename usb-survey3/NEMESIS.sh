#!/system/bin/sh
# Read-only copy of theme-related files. Writes only to the USB drive.
export PATH=/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin
O=/storage/usb0/THEME
mkdir $O
I=$O/info.txt
echo "copy start" > $I
for f in /system/app/ThemeSettings.apk /system/app/ThemeSettings.odex \
         /system/app/HKMC_HomeSetting_WP.apk /system/app/HKMC_HomeSetting_WP.odex \
         /system/framework/hkmc-res.apk \
         /system/etc/permissions/com.hkmc.software.theme.settings.xml; do
  cp $f $O/ >> $I 2>&1
  echo "$f -> $?" >> $I
done
ls -lR /system > $O/system-tree.txt 2>&1
ls -l $O >> $I 2>&1
echo "copy done" >> $I
sync
sleep 3
sync
