      subroutine cardiac_simulation(Lx, Ly, nstim, iseed, rbcl, Dfu,
     &     gicai, gtos, gtof, gnacai, zxr, nbt, cxinit, xnai, xnao,
     &     xki, xko, cao, temp, xxr, xf, dt, v_out, cb_out, csrb_out,
     &     ci_out, t_out, num_steps)
      
      implicit none
      
C     Input parameters
      integer, intent(in) :: Lx, Ly, nstim, iseed, nbt
      double precision, intent(in) :: rbcl, Dfu
      double precision, intent(in) :: gicai, gtos, gtof, gnacai, zxr
      double precision, intent(in) :: cxinit, xnai, xnao, xki, xko
      double precision, intent(in) :: cao, temp, xxr, xf, dt
      
C     Output arrays - break these into multiple lines
      integer, intent(out) :: num_steps
      integer :: steps_per_output
      steps_per_output = nstim*int(rbcl/dt)/100+1
      
      double precision, intent(out) :: v_out(Lx,Ly,steps_per_output)
      double precision, intent(out) :: cb_out(Lx,Ly,steps_per_output)
      double precision, intent(out) :: csrb_out(Lx,Ly,steps_per_output)
      double precision, intent(out) :: ci_out(Lx,Ly,steps_per_output)
      double precision, intent(out) :: t_out(steps_per_output)
      
C     Local arrays - break these into multiple lines
      double precision, allocatable :: v(:,:)
      double precision, allocatable :: vnew(:,:)
      double precision, allocatable :: dv(:,:)
      double precision, allocatable :: vold(:,:)
      double precision, allocatable :: xm(:,:)
      double precision, allocatable :: xh(:,:)
      double precision, allocatable :: xj(:,:)
      double precision, allocatable :: xr(:,:)
      double precision, allocatable :: xs1(:,:)
      double precision, allocatable :: qks(:,:)
      double precision, allocatable :: xkur(:,:)
      double precision, allocatable :: ykur(:,:)
      double precision, allocatable :: xtof(:,:)
      double precision, allocatable :: ytof(:,:)
      double precision, allocatable :: xtos(:,:)
      double precision, allocatable :: ytos(:,:)
      double precision, allocatable :: cb(:,:)
      double precision, allocatable :: ci(:,:)
      double precision, allocatable :: csrb(:,:)
      double precision, allocatable :: csri(:,:)
      
C     Local variables
      double precision :: pbxi, time, frt, dx, dy, duration
      integer :: ix, iy, nstep, output_count, iz, ncount, mstp
      
C     Allocate arrays
      allocate(v(0:Lx+1,0:Ly+1))
      allocate(vnew(0:Lx+1,0:Ly+1))
      allocate(dv(0:Lx,0:Ly))
      allocate(vold(0:Lx,0:Ly))
      allocate(xm(0:Lx,0:Ly))
      allocate(xh(0:Lx,0:Ly))
      allocate(xj(0:Lx,0:Ly))
      allocate(xr(0:Lx,0:Ly))
      allocate(xs1(0:Lx,0:Ly))
      allocate(qks(0:Lx,0:Ly))
      allocate(xkur(0:Lx,0:Ly))
      allocate(ykur(0:Lx,0:Ly))
      allocate(xtof(0:Lx,0:Ly))
      allocate(ytof(0:Lx,0:Ly))
      allocate(xtos(0:Lx,0:Ly))
      allocate(ytos(0:Lx,0:Ly))
      allocate(cb(0:Lx,0:Ly))
      allocate(ci(0:Lx,0:Ly))
      allocate(csrb(0:Lx,0:Ly))
      allocate(csri(0:Lx,0:Ly))
      
C     Initialize variables
      pbxi = 0.5d0  
      time = 0.0d0  
      
C     ... rest of your code would go here
      
C     Deallocate before exiting
      deallocate(v)
      deallocate(vnew)
      deallocate(dv)
      deallocate(vold)
      deallocate(xm)
      deallocate(xh)
      deallocate(xj)
      deallocate(xr)
      deallocate(xs1)
      deallocate(qks)
      deallocate(xkur)
      deallocate(ykur)
      deallocate(xtof)
      deallocate(ytof)
      deallocate(xtos)
      deallocate(ytos)
      deallocate(cb)
      deallocate(ci)
      deallocate(csrb)
      deallocate(csri)
      
      end subroutine cardiac_simulation
