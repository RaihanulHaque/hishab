#!/bin/zsh
# Send test messages through the Hishab app on the Pixel, safely.
#   zsh app_messages.sh "5 taka rickshaw bhara" "cha 8 tk ar singara 5 tk" ...
# Every tap is preceded by a screen check (Home: balance card + "Recent", no edit/review sheet);
# any unexpected screen aborts with a screenshot and touches nothing. Review sheets are closed with
# Back, never saved. Messages must be ASCII (adb input text cannot type Bangla script).
S=${ADB_SERIAL:-38041FDJH00B57}
OUT=${OUT_DIR:-/tmp}

dump() { adb -s $S shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1; adb -s $S exec-out cat /sdcard/ui.xml; }
is_home() {
  local x; x=$(dump)
  [[ "$x" == *'Balance'* && "$x" == *'content-desc="Recent"'* && "$x" != *'Edit transaction'* && "$x" != *'Found '* ]]
}
composer_bounds() { dump | tr '>' '\n' | grep 'android.widget.EditText' | grep -o 'bounds="[^"]*"' | head -1 | tr -dc '0-9,[]' ; }
composer_text() { dump | tr '>' '\n' | grep 'android.widget.EditText' | grep -o 'text="[^"]*"' | head -1 | sed 's/text="//;s/"$//'; }
sheet_items() { dump | tr '>' '\n' | grep -o 'content-desc="[^"]*"' | grep -E '৳' | grep -v Balance | sed 's/content-desc="//;s/"$//;s/&#10;/ · /g'; }
abort() { echo "ABORT: $1"; adb -s $S exec-out screencap -p > $OUT/hishab_abort.png; exit 1; }

clear_composer() {
  adb -s $S shell input keyevent $(printf '112 %.0s' $(seq 1 80))
  adb -s $S shell input keyevent $(printf '67 %.0s' $(seq 1 80))
}

send() {
  local msg="$1" n="$2"
  is_home || abort "not on Home before tapping the composer (#$n)"
  # composer box [x1,y1][x2,y2]; tap its text area, the send button sits at its right end
  local b; b=$(composer_bounds); [[ -n "$b" ]] || abort "composer not found (#$n)"
  local y1 y2; y1=$(echo $b | sed -E 's/\[[0-9]+,([0-9]+)\].*/\1/'); y2=$(echo $b | sed -E 's/.*\[[0-9]+,([0-9]+)\]$/\1/')
  local ymid=$(( (y1 + y2) / 2 ))
  adb -s $S shell input tap 400 $ymid; sleep 1
  is_home || abort "screen changed after tapping the composer (#$n)"
  clear_composer
  [[ -z "$(composer_text)" ]] || abort "composer not empty after clearing (#$n)"
  adb -s $S shell input text "${msg// /%s}"; sleep 1
  [[ "$(composer_text)" == "$msg" ]] || abort "composer holds '$(composer_text)', expected '$msg' (#$n)"
  is_home || abort "not on Home before send (#$n)"
  b=$(composer_bounds); y1=$(echo $b | sed -E 's/\[[0-9]+,([0-9]+)\].*/\1/'); y2=$(echo $b | sed -E 's/.*\[[0-9]+,([0-9]+)\]$/\1/')
  adb -s $S logcat -c
  adb -s $S shell input tap 967 $(( (y1 + y2) / 2 ))
  for k in $(seq 1 60); do adb -s $S logcat -d -s flutter | grep -q "hishab:" && break; sleep 1; done
  sleep 2
  echo "#$n  $msg  =>  $(adb -s $S logcat -d -s flutter | grep 'hishab:' | sed 's/.*hishab: //')"
  local x; x=$(dump)
  if [[ "$x" == *'Found '* ]]; then
    sheet_items | sed 's/^/      /'
    adb -s $S exec-out screencap -p > $OUT/hishab_msg_$n.png
    adb -s $S shell input keyevent KEYCODE_BACK; sleep 1
  else
    echo "      (no review sheet)"
  fi
  is_home || abort "not back on Home after closing the sheet (#$n)"
  sleep 2
}

n=1
for m in "$@"; do send "$m" $n; n=$((n + 1)); done
# leave the composer empty and the keyboard closed
if is_home; then
  b=$(composer_bounds); y1=$(echo $b | sed -E 's/\[[0-9]+,([0-9]+)\].*/\1/'); y2=$(echo $b | sed -E 's/.*\[[0-9]+,([0-9]+)\]$/\1/')
  adb -s $S shell input tap 400 $(( (y1 + y2) / 2 )); sleep 1
  is_home && { clear_composer; adb -s $S shell input keyevent KEYCODE_BACK; }
fi
echo "composer now: '$(composer_text)'"
