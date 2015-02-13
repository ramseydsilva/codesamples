using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Data.Entity;
using System.Linq;
using WebMatrix.WebData;
using System.Data.Entity.ModelConfiguration.Conventions;
using Gnet.Areas.Ftvcdl.Models;

namespace Gnet.Models
{
    [Table("UserGroups")]
    public class Group
    {   
        public int id { get; set; }
        public string Name { get; set; }

        public virtual ICollection<UserProfile> Users { get; set; }

    }

    [Table("UserProfile")]
    public class UserProfile
    {
        [Key]
        [DatabaseGeneratedAttribute(DatabaseGeneratedOption.Identity)]
        public int UserId { get; set; }
        public string UserName { get; set; }
        public string FullName { get; set; }

        public bool is_superadmin { get; set; }
        public bool is_methods_superadmin { get; set; }
        public bool is_baex_superadmin { get; set; }
        public bool is_logistics_superadmin { get; set; }
        public bool is_mi_superadmin { get; set; }
        public bool is_staff { get; set; }
        public bool is_authorized { get; set; }
        public bool is_ftvcdl_authorized { get; set; }
        public DateTime? access_requested_date { get; set; }

        public string custom_columns { get; set; }

        public virtual ICollection<Group> Groups { get; set; }

        public virtual ICollection<PartBase> MethodsResponsiblePartBases { get; set; }
        public virtual ICollection<PartBase> MiResponsiblePartBases { get; set; }
        public virtual ICollection<PartBase> LogisticsResponsiblePartBases { get; set; }
        public virtual ICollection<PartBase> BaexResponsiblePartBases { get; set; }
        public virtual ICollection<PartBase> SuperAdminPartBases { get; set; }

        public virtual ICollection<WorkPack> MethodsResponsibleWorkPacks { get; set; }
        public virtual ICollection<WorkPack> BaexResponsibleWorkPacks { get; set; }
        public virtual ICollection<WorkPack> MiResponsibleWorkPacks { get; set; }
        public virtual ICollection<WorkPack> LogisticsResponsibleWorkPacks { get; set; }
        public virtual ICollection<WorkPack> SuperAdminWorkPacks { get; set; }

        public virtual ICollection<WorkPackShipTo> MethodsResponsibleWorkPackShipTos { get; set; }
        public virtual ICollection<WorkPackShipTo> BaexResponsibleWorkPackShipTos { get; set; }
        public virtual ICollection<WorkPackShipTo> MiResponsibleWorkPackShipTos { get; set; }
        public virtual ICollection<WorkPackShipTo> LogisticsResponsibleWorkPackShipTos { get; set; }
        public virtual ICollection<WorkPackShipTo> SuperAdminWorkPackShipTos { get; set; }

        public UserProfile()
        {
            is_superadmin = false;
            is_methods_superadmin = false;
            is_baex_superadmin = false;
            is_logistics_superadmin = false;
            is_mi_superadmin = false;
            is_staff = false;
            is_authorized = false;
            is_ftvcdl_authorized = false;
            custom_columns = "[]";
        }

        public string DisplayName()
        {
            if (this.FullName != null)
            {
                return this.FullName;
            }
            else
            {
                return this.UserName;
            }
        }

        public string ResponsibleWorkPacks(ICollection<WorkPackShipTo> ResponsibleWorkPackShipTos)
        {
            List<string> wps = new List<string>();
            foreach (WorkPackShipTo wpst in ResponsibleWorkPackShipTos)
            {
                if (!wps.Contains(wpst.work_pack.name))
                {
                    wps.Add(wpst.work_pack.name);
                }
            }
            return String.Join(", ", wps);
        }

        public Dictionary<string, string> dict()
        {

            Dictionary<string, string> user = new Dictionary<string, string>();
            user.Add("id", this.UserId.ToString());
            user.Add("display_name", this.DisplayName());
            user.Add("is_superadmin", this.is_superadmin.ToString());
            user.Add("is_methods_superadmin", this.is_methods_superadmin.ToString());
            user.Add("is_baex_superadmin", this.is_baex_superadmin.ToString());
            user.Add("is_logistics_superadmin", this.is_logistics_superadmin.ToString());
            user.Add("is_mi_superadmin", this.is_mi_superadmin.ToString());
            user.Add("is_staff", this.is_staff.ToString());

            return user;
        }
    }

    public class UserContext : DbContext
    {
        public DbSet<UserProfile> Users { get; set; }
        public DbSet<Group> Groups { get; set;  }

        public UserProfile GetOrCreateUserProfile(string UserName, string FullName)
        {
            IQueryable<UserProfile> users = Users.Where(w => w.UserName == UserName);

            if (users.Count() > 0)
            {
                UserProfile user = users.First();
                return user;
            }
            else
            {
                // check if full name exists

                users = Users.Where(w => w.FullName == FullName);
                if (users.Count() > 0)
                {
                    UserProfile user = users.First();
                    return user;
                }

                // Username and fullnamem does not exist so create new user
                WebSecurity.CreateUserAndAccount(UserName, "blankpass");
                UserProfile profile = this.Users.Single(u => u.UserName == UserName);

                profile.UserName = UserName;
                profile.FullName = FullName;

                this.SaveChanges();

                return profile;
            }
        }

        protected override void OnModelCreating(DbModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Conventions.Remove<OneToManyCascadeDeleteConvention>();

            modelBuilder.Entity<Group>()
                .HasMany(m => m.Users)
                .WithMany(t => t.Groups)
                .Map(m =>
                {
                    m.ToTable("Group_Users_Mapping");
                    m.MapLeftKey("GroupID");
                    m.MapRightKey("UserProfileID");
                }
            );

        }
    }
}