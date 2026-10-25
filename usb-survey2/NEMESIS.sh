#!/system/bin/sh
# Read-only copy of camera-related files. Writes only to the USB drive.
export PATH=/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin
O=/storage/usb0/CAMERA
mkdir $O
I=$O/info.txt
echo "copy start" > $I
for f in /system/app/HKMC_RearCamera.apk /system/app/HKMC_RearCamera.odex \
         /system/app/HKMC_RearCareMode.apk /system/app/HKMC_RearCareMode.odex \
         /system/onebin/neev23_us/featurexml.cfg /system/onebin/neev23_us/mobis.prop \
         /system/onebin/neev23_us/pkg.cfg \
         /system/etc/permissions/com.hkmc.software.rearcamera.maintain.xml \
         /system/etc/permissions/com.hkmc.software.regulation.xml; do
  cp $f $O/ >> $I 2>&1
  echo "$f -> $?" >> $I
done
ls -l $O >> $I 2>&1
echo "copy done" >> $I
