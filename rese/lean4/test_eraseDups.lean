import Std
import Mathlib.Data.List.Dedup

-- Check what's available about eraseDups
#check @List.eraseDups

-- Check if there's a sublist theorem for eraseDups
#check List.Sublist

-- Check Mathlib's dedup
#check @List.dedup
#check List.dedup_sublist

-- Check what sublist operations are available
#check Sublist.length_le
