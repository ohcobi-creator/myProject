#!/system/bin/sh
# Read-only survey. Writes only to the USB drive (/storage/usb0/SURVEY).
export PATH=/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin
O=/storage/usb0/SURVEY
mkdir $O
I=$O/info.txt
run() {
  echo "### $*" >> $I
  "$@" >> $I 2>&1
  echo "" >> $I
}
echo "survey start" > $I
for k in ro.build.product2 ro.product.device ro.build.version.sdk ro.product.vehicleinfo \
         ro.product.vcrm.model ro.product.opt.stepofspec ro.product.opt.vehicletype \
         ro.product.opt.hud.artist ro.mobis.targetdir ro.sf.lcd_density; do
  echo "$k=$(getprop $k)" >> $I
done
echo "" >> $I
run ls -l /system/app
run ls -l /system/priv-app
run ls -l /system/framework
run ls -l /system/etc/permissions
run ls -lR /system/onebin/neev23_us
run ls -l /dev
run service list
run pm list packages -f
echo "survey done" >> $I
