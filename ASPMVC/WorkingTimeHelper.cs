using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Gnet.Areas.MaterialMaster.Utility
{
    public class WorkingTime
    {
        public static bool IsPublicHoliday(DateTime date)
        {
            return false; // TODOB: TBD
        }

        public static bool IsWorkingDay(DateTime date)
        {
            if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday || IsPublicHoliday(date))
                return false;
            return true;
        }

        public static long GetWorkingTicks(DateTime start_date, DateTime end_date)
        {
            DateTime ClockOutTime = new DateTime(start_date.Year, start_date.Month, start_date.Day, 15, 0, 0, 0);
            DateTime ClockInTime = new DateTime(start_date.Year, start_date.Month, start_date.Day, 7, 0, 0, 0);           

            if (start_date > ClockOutTime) // If start_date is after clock out time of the same day
            {
                return GetWorkingTicks(ClockInTime.AddDays(1), end_date);
            } 
            else if (start_date < ClockInTime) // If start_date is before clock out time of the same day
            {
                return GetWorkingTicks(ClockInTime, end_date);
            } 
            else if (start_date < end_date)
            {
                if (IsWorkingDay(start_date))
                {

                    if (start_date.DayOfYear == end_date.DayOfYear) // If start_date is between 7AM and 3PM of the same day
                    {
                        if (end_date.DayOfYear == ClockOutTime.DayOfYear && end_date > ClockOutTime) // If end date is after 3PM
                            end_date = new DateTime(end_date.Year, end_date.Month, end_date.Day, 15, 0, 0, 0); // then set end date to be 3PM
                        return (end_date - start_date).Ticks;
                    }
                    else  // If start_date is between 7AM and 3PM of different days
                    {
                        return (ClockOutTime - start_date).Ticks + GetWorkingTicks(ClockInTime.AddDays(1), end_date);
                    }
                }
                else
                {
                    return GetWorkingTicks(ClockInTime.AddDays(1), end_date); // If start_date is on non working day, push to begin of next day
                }
            }

            // else if (start_date >= end_date)
            return 0;
        }

        public static double GetWorkingSeconds(DateTime start_date, DateTime end_date)
        {
            return TimeSpan.FromTicks(GetWorkingTicks(start_date, end_date)).TotalSeconds;
        }

        public static double GetWorkingMinutes(DateTime start_date, DateTime end_date)
        {
            return TimeSpan.FromTicks(GetWorkingTicks(start_date, end_date)).TotalMinutes;
        }

        public static double GetWorkingHours(DateTime start_date, DateTime end_date)
        {
            return TimeSpan.FromTicks(GetWorkingTicks(start_date, end_date)).TotalHours;
        }

        public static double GetWorkingDays(DateTime start_date, DateTime end_date)
        {
            return TimeSpan.FromTicks(GetWorkingTicks(start_date, end_date)).TotalDays;
        }
    }
}