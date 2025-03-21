subroutine cardiac_simulation(Lx, Ly, nstim, iseed, rbcl, Dfu, &
                             gicai, gtos, gtof, gnacai, zxr, &
                             nbt, cxinit, xnai, &
                             xnao, xki, xko, cao, temp, xxr, xf, dt, &
                             max_buffer_size,mod_output, parallel_exec, &
                             v_out, cb_out, csrb_out, ci_out, t_out, num_steps)

    implicit double precision (a-h,o-z)
    

    ! Input parameters
    integer, intent(in) :: Lx, Ly, nstim, iseed, nbt
    real(kind=8), intent(in) :: rbcl, Dfu
    real(kind=8), intent(in) :: gicai, gtos, gtof, gnacai, zxr
    real(kind=8), intent(in) :: cxinit, xnai, xnao, xki, xko
    real(kind=8), intent(in) :: cao, temp, xxr, xf, dt
    integer, intent(in) :: max_buffer_size  ! New parameter for array sizes
    integer, intent(in) :: mod_output
    logical, intent(in) :: parallel_exec


    
    ! Output parameters
    integer, intent(out) :: num_steps
    
    ! Output arrays with dimensions controlled by Python
    real(kind=8), intent(out) :: v_out(max_buffer_size,Lx,Ly)
    real(kind=8), intent(out) :: cb_out(max_buffer_size,Lx,Ly)
    real(kind=8), intent(out) :: csrb_out(max_buffer_size,Lx,Ly)
    real(kind=8), intent(out) :: ci_out(max_buffer_size,Lx,Ly)
    real(kind=8), intent(out) :: t_out(max_buffer_size)
    
    ! Rest of your code remains the same...

    
    ! Local arrays
    real(kind=8), allocatable :: v(:,:), vnew(:,:), dv(:,:), vold(:,:)
    real(kind=8), allocatable :: xm(:,:), xh(:,:), xj(:,:), xr(:,:)
    real(kind=8), allocatable :: xs1(:,:), qks(:,:), xkur(:,:), ykur(:,:)
    real(kind=8), allocatable :: xtof(:,:), ytof(:,:), xtos(:,:), ytos(:,:)
    real(kind=8), allocatable :: cb(:,:), ci(:,:), csrb(:,:), csri(:,:)
    
    ! Additional arrays from the original file
    real(kind=8), allocatable :: cnsr(:,:), po(:,:), c1(:,:), c2(:,:)
    real(kind=8), allocatable :: xi1(:,:), xi2(:,:), xi2s(:,:), c1s(:,:)
    real(kind=8), allocatable :: c2s(:,:), xi1s(:,:), cit(:,:), cbt(:,:)
    real(kind=8), allocatable :: xicaqz(:,:), xinacaqz(:,:)
    real(kind=8), allocatable :: xitor(:,:), xitorf(:,:), zina(:,:)
    real(kind=8), allocatable :: zica(:,:)
    real(kind=8), allocatable :: pi(:,:), pb(:,:), pox(:,:), pos(:,:)
    real(kind=8), allocatable :: ra(:,:)
    integer, allocatable :: nsb(:,:), nsi(:,:)
    real(kind=8), allocatable :: ctmax(:)
    real(kind=8), allocatable :: camp(:,:), cmax(:,:,:)
    real(kind=8), allocatable :: uxx(:,:), xuu(:,:)
    real(kind=8), allocatable :: gicaz(:,:), gnacaz(:,:)
    real(kind=8), allocatable :: pbxz(:,:)
    
    ! Local variables
    real(kind=8) :: pbxi, time, frt, dx, dy, duration
    integer :: ix, iy, nstep, output_count, iz, ncount, mstp

    !f2py intent(callback) stimulus_function
    external stimulus_function
    real(kind=8) :: stimulus_function


    
    ! Allocate local arrays with proper dimensions
    allocate(v(0:Lx+1,0:Ly+1))
    allocate(vnew(0:Lx+1,0:Ly+1))
    allocate(dv(0:Lx,0:Ly))
    allocate(vold(0:Lx,0:Ly))
    
    ! Match dimensions from original file (Lx,Ly)
    allocate(xm(Lx,Ly))
    allocate(xh(Lx,Ly))
    allocate(xj(Lx,Ly))
    allocate(xr(Lx,Ly))
    allocate(xs1(Lx,Ly))
    allocate(qks(Lx,Ly))
    allocate(xkur(Lx,Ly))
    allocate(ykur(Lx,Ly))
    allocate(xtof(Lx,Ly))
    allocate(ytof(Lx,Ly))
    allocate(xtos(Lx,Ly))
    allocate(ytos(Lx,Ly))
    allocate(cb(Lx,Ly))
    allocate(ci(Lx,Ly))
    allocate(csrb(Lx,Ly))
    allocate(csri(Lx,Ly))
    
    ! Allocate additional arrays from original file
    allocate(cnsr(Lx,Ly))
    allocate(po(Lx,Ly))
    allocate(c1(Lx,Ly))
    allocate(c2(Lx,Ly))
    allocate(xi1(Lx,Ly))
    allocate(xi2(Lx,Ly))
    allocate(xi2s(Lx,Ly))
    allocate(c1s(Lx,Ly))
    allocate(c2s(Lx,Ly))
    allocate(xi1s(Lx,Ly))
    allocate(cit(Lx,Ly))
    allocate(cbt(Lx,Ly))
    allocate(xicaqz(Lx,Ly))
    allocate(xinacaqz(Lx,Ly))
    allocate(xitor(Lx,Ly))
    allocate(xitorf(Lx,Ly))
    allocate(zina(Lx,Ly))
    allocate(zica(Lx,Ly))
    allocate(pi(Lx,Ly))
    allocate(pb(Lx,Ly))
    allocate(pox(Lx,Ly))
    allocate(pos(Lx,Ly))
    allocate(ra(Lx,Ly))
    allocate(nsb(Lx,Ly))
    allocate(nsi(Lx,Ly))
    
    ! Match dimensions for the remaining arrays
    allocate(ctmax(1000))
    allocate(camp(0:Lx,0:Ly))
    allocate(cmax(0:Lx,0:Ly,0:1000))
    allocate(uxx(0:Lx,0:Ly))
    allocate(xuu(0:Lx,0:Ly))
    allocate(gicaz(0:Lx,0:Ly))
    allocate(gnacaz(0:Lx,0:Ly))
    allocate(pbxz(0:Lx,0:Ly))





    
    ! Initialize variables
    pbxi = 0.5d0  
    time = 0.0d0  
    
    ! ... rest of your code would go here
    ! Initial Value Initialization: 

    ! Calculate derived parameters
    frt = xf/(xxr*temp)
    dx = 0.015d0
    dy = 0.015d0
    duration = rbcl
    nstep = duration/dt
    slmbdax = Dfu*dt/4.0d0/dx/dx
    slmbday = Dfu*dt/4.0d0/dy/dy


     ! Initialize the state variables across the grid
    do ix = 1, Lx
        do iy = 1, Ly
            
            ! Ca variables
            cb(ix,iy) = 0.1d0  ! boundary cytosolic Ca
            ci(ix,iy) = 0.1d0  ! interior cytosolic Ca
            
            csrb(ix,iy) = cxinit  ! boundary SR Ca
            csri(ix,iy) = cxinit  ! interior SR Ca
            
            po(ix,iy) = 0.0d0  ! LCC model facing cb 
            c1(ix,iy) = 0.0d0  
            c2(ix,iy) = 1.0d0
            xi1(ix,iy) = 0.0d0
            xi2(ix,iy) = 0.0d0
            
            pos(ix,iy) = 0.0d0  ! LCC model facing spark
            c1s(ix,iy) = 0.0d0
            c2s(ix,iy) = 0.0d0
            xi1s(ix,iy) = 0.0d0
            xi2s(ix,iy) = 0.0d0
            
            call total(ci(ix,iy), cit(ix,iy))
            call total(cb(ix,iy), cbt(ix,iy))
            
            nsb(ix,iy) = 5
            
            v(ix,iy) = -90.0d0
            xm(ix,iy) = 0.001d0 ! Ina gate variable
            xh(ix,iy) = 1.0d0   ! Ina gate variable
            xj(ix,iy) = 1.0d0   ! Ina gate variable
            xr(ix,iy) = 0.0d0
            xs1(ix,iy) = 0.3d0 ! iks gate var. 
            qks(ix,iy) = 0.2d0 ! Ca gating of iks
            
            xtos(ix,iy) = 0.01d0 ! ito activation
            ytos(ix,iy) = 0.9d0 ! ito inactivation
            xtof(ix,iy) = 0.02d0 ! ito fast activation
            ytof(ix,iy) = 0.8d0  ! ito slow inactivation
            
            vold(ix,iy) = v(ix,iy)
            
            gicaz(ix,iy) = gicai
            gnacaz(ix,iy) = gnacai
            pbxz(ix,iy) = pbxi
            
        enddo
    enddo

    ! store Initial State in Output parameters: 
    output_count = 1
    do ix = 1, Lx
        do iy = 1, Ly
            v_out(output_count,ix,iy) = v(ix,iy)
            cb_out(output_count,ix,iy) = cb(ix,iy)
            csrb_out(output_count,ix,iy) = csrb(ix,iy)
            ci_out(output_count,ix,iy) = ci(ix,iy)
        enddo
    enddo

    ! Integration Loop
	kku=1
	sapd=0.0d0
	sapd2=0.0d0

	do iz=1,nstim  ! number of beats 

	nstep=int(rbcl/dt)

	umax=0.0
	vmax=0.0
	

       do ncount = 0, nstep


	 time = dfloat(ncount)*dt ! time during each beat
       
          do  iy =1, Ly
            do  ix=1,Lx

! ********************* heterogeneity in tissue *********************

	gica=gicaz(ix,iy)
	gnaca=gnacaz(ix,iy)
	pbx=pbxz(ix,iy)

! ********************************************************************

	call xfree(cit(ix,iy),ci(ix,iy))  ! convert total Ca to free
	call xfree(cbt(ix,iy),cb(ix,iy))  ! convert total Ca to free

! ******* fraction of clusters with sparks ***************************

	pb(ix,iy)=dfloat(nsb(ix,iy))/dfloat(nbt)

!*********************************************************************

	vupb=0.4
	vupi=0.4

        call uptake(cb(ix,iy),vupb,xupb)  ! uptake at boundary 
	call uptake(ci(ix,iy),vupi,xupi)  ! uptake at interior sites

! *******************************************************************************

	call inaca(v(ix,iy),frt,xnai,xnao,cao,cb(ix,iy),xinacaq1)

	xinacaq=gnaca*xinacaq1
	
	xinacaqz(ix,iy)=xinacaq

	pox(ix,iy)=po(ix,iy)+pos(ix,iy)
	call ica(v(ix,iy),frt,cao,cb(ix,iy),pox(ix,iy),rca,xicaq) ! LCC current
    
	xicaq=gica*130.0*xicaq

! ******************************************************************************* 	

	! spark rate at junctional dyads
	
	qq=0.5d0
	ab=35.0*qq
	csrx=600.0d0
	phisr=1.0/(1.0+(csrx/csrb(ix,iy))**10) ! cutoff rate bellow 500 muM

	alphab=ab*dabs(rca)*po(ix,iy)*phisr ! spark on rate due to LCC 
	bts=1.0/30.0   ! spark off rate 

	call markov(dt,v(ix,iy),cb(ix,iy),c1(ix,iy),&
       c2(ix,iy),xi1(ix,iy),xi2(ix,iy),po(ix,iy),&
       c1s(ix,iy),c2s(ix,iy),&
       xi1s(ix,iy), xi2s(ix,iy),pos(ix,iy),alphab,bts,zxr)

! *****************************************************************************

	gsrb=(0.01/1.5)*1.0
	xryrb=gsrb*csrb(ix,iy)*pb(ix,iy)    ! ryr current for boundary region

! ******************************************************************************
	! spark rate in the interior
	
	xryri=0.0d0

!*****************************************************************************	

	vi=0.50  ! volume factors
	vb=1.0

	vbi=vb/vi ! ratio of vb volume to vi 
	vbisr=vbi ! ratio of sr volume for nj to j

	vq=30.0   ! volume factors
	
	vbsr=vq  ! this is the ratio of vb/vsrb  is set to 30	
	visr=vq  ! this is the ratio of vi/vsri

	tau1=5.0  ! diffusive timescale
	tau2=5.0  ! diffusive timescale 

	dfbi=(cb(ix,iy)-ci(ix,iy))/tau1  ! the diffusive currents
	dfbisr=(csrb(ix,iy)-csri(ix,iy))/tau2

	xsarc=-xicaq+xinacaq ! boundary currents

	
	dcbt=xryrb-xupb+xsarc-dfbi
	dcsrb=vbsr*(-xryrb+xupb)-dfbisr

	dcit=xryri-xupi+vbi*dfbi
        dcsri=visr*(-xryri+xupi)+vbisr*dfbisr
        
	cbt(ix,iy)=cbt(ix,iy)+dcbt*dt
	cit(ix,iy)=cit(ix,iy)+dcit*dt
	
	csrb(ix,iy)=csrb(ix,iy)+dcsrb*dt	
	csri(ix,iy)=csri(ix,iy)+dcsri*dt


! ********* time evolution due to binomial distribution *****************

	nsbx=nsb(ix,iy)
	call binevol(nbt,nsbx,alphab,bts,dt,iseed,ndeltapx,ndeltamx)

	if(ndeltamx.gt.nsbx.or.ndeltapx.gt.nbt) then
	nsb(ix,iy)=0
	else
	nsb(ix,iy)=nsb(ix,iy)+ndeltapx-ndeltamx
	endif 


!  ********** do voltage here with adaptive time step ********************
	
	 ! pass currents from Ca system to voltage 

	 wca=12.0d0
         xinaca=wca*xinacaq  ! convert ion flow to current
         xica=2.0d0*wca*xicaq
         
         zica(ix,iy)=xica

! -------------  time step adjustment ------------------------

                adq=dabs(dv(ix,iy))
                
                if(adq.gt.25.0d0) then ! finer time step when dv/dt large
                mstp=10
                else
                mstp=1
                endif 
	              
		hode=dt/dfloat(mstp) 

                do iii=1 , mstp  

!************ these are the voltage dependent currents ***********************
    
	call ina(hode,v(ix,iy),frt,xh(ix,iy),xj(ix,iy),xm(ix,iy),xnai,xnao,xina) ! sodium
          
	zina(ix,iy)=xina


	call ikr(hode,v(ix,iy),frt,xko,xki,xr(ix,iy),xikr) ! ikr

      	call iks(hode,v(ix,iy),frt,cb(ix,iy),xnao,xnai,xko,xki,xs1(ix,iy),qks(ix,iy),xiks) ! iks

        call ik1(hode,v(ix,iy),frt,xki,xko,xik1)  ! ik1

      	call ito(hode,v(ix,iy),frt,xki,xko,xtof(ix,iy),ytof(ix,iy),xtos(ix,iy),ytos(ix,iy),xito,gtof,gtos) ! ito

        call inak(v(ix,iy),frt,xko,xnao,xnai,xinak) ! inak

	xitor(ix,iy)=xito

!**********UNIFORM STIMULATION *************************************
          
	!if(time.lt.1.0) then
	!stim=80.0d0
	!else
	!stim=0.0
	!endif       
!       EDGE STIMULATION 
    stim = stimulus_function(ix, iy, time) !Use python callback function


    !    if(time.lt.1.0) then 
    !      if(ix.lt.10 .and. iy.lt.10) then ! stim corrner
    !         stim = 80.0d0
    !      else
    !         stim = 0.0
    !      endif
    !      else
    !        stim = 0.0
    !      endif

!*************************************************************************
	
        dvh=-(xina+xik1+xikr+xiks+xito+xinaca+xica+xinak)+ stim

	

        v(ix,iy)=v(ix,iy)+dvh*hode

	         enddo 

             end do

        end do


                  ! Apply diffusion with Euler Forward
            if (parallel_exec) then 
                call Euler_forward_optimized(Lx, Ly, v, vnew, slmbdax, slmbday)
                call Euler_forward_optimized(Lx, Ly, v, vnew, slmbdax, slmbday)
            else 
                call Euler_forward(Lx, Ly, v, vnew, slmbdax, slmbday)
                call Euler_forward(Lx, Ly, v, vnew, slmbdax, slmbday)
            endif

                  
                  ! Store output at desired intervals
                  if (mod(ncount, mod_output) .eq. 0) then
                      do ix = 1, Lx
                          do iy = 1, Ly
                              v_out(output_count,ix,iy) = v(ix,iy)
                              cb_out(output_count,ix,iy) = cb(ix,iy)
                              csrb_out(output_count,ix,iy) = csrb(ix,iy)
                              ci_out(output_count,ix,iy) = ci(ix,iy)
                          enddo
                      enddo
                      t_out(output_count) = t
                      output_count = output_count + 1
                  endif
                  
              enddo ! time step
          enddo ! beats
          
          ! Record final number of output steps
          num_steps = output_count - 1
          
          return

    
    ! Deallocate all arrays before exiting
    deallocate(v, vnew, dv, vold)
    deallocate(xm, xh, xj, xr)
    deallocate(xs1, qks, xkur, ykur)
    deallocate(xtof, ytof, xtos, ytos)
    deallocate(cb, ci, csrb, csri)
    deallocate(cnsr, po, c1, c2)
    deallocate(xi1, xi2, xi2s, c1s)
    deallocate(c2s, xi1s, cit, cbt)
    deallocate(xicaqz, xinacaqz)
    deallocate(xitor, xitorf, zina)
    deallocate(zica)
    deallocate(pi, pb, pox, pos)
    deallocate(ra)
    deallocate(nsb, nsi)
    deallocate(ctmax)
    deallocate(camp, cmax)
    deallocate(uxx, xuu)
    deallocate(gicaz, gnacaz)
    deallocate(pbxz)
    
    ! Note: You don't need to deallocate the output arrays
    ! They will be automatically deallocated by the calling program
    
end subroutine cardiac_simulation


subroutine Euler_forward(LLx, LLy, v, vnew, slmbdax, slmbday)
    implicit double precision (a-h, o-z)
    
    double precision, dimension(0:LLx+1, 0:LLy+1) :: v, vnew
    
    ! Set corner boundary conditions
    v(0, 0) = v(2, 2)
    v(0, LLy+1) = v(2, LLy-1)
    v(LLx+1, 0) = v(LLx-1, 2)
    v(LLx+1, LLy+1) = v(LLx-1, LLy-1)
    
    ! Set non-flux boundary conditions for x-direction
    do ix = 1, LLx
        v(ix, 0) = v(ix, 2)
        v(ix, LLy+1) = v(ix, LLy-1)
    end do
    
    ! Set non-flux boundary conditions for y-direction
    do iy = 1, LLy
        v(0, iy) = v(2, iy)
        v(LLx+1, iy) = v(LLx-1, iy)
    end do
    
    ! First update step
    do ix = 1, LLx
        do iy = 1, LLy
            vnew(ix, iy) = v(ix, iy) + slmbdax * (v(ix+1, iy) + v(ix-1, iy) - 2.0d0 * v(ix, iy)) &
                                     + slmbday * (v(ix, iy+1) + v(ix, iy-1) - 2.0d0 * v(ix, iy))
        end do
    end do
    
    ! Update boundary conditions for vnew
    vnew(0, 0) = vnew(2, 2)
    vnew(0, LLy+1) = vnew(2, LLy-1)
    vnew(LLx+1, 0) = vnew(LLx-1, 2)
    vnew(LLx+1, LLy+1) = vnew(LLx-1, LLy-1)
    
    ! Update non-flux boundary conditions for x-direction for vnew
    do ix = 1, LLx
        vnew(ix, 0) = vnew(ix, 2)
        vnew(ix, LLy+1) = vnew(ix, LLy-1)
    end do
    
    ! Update non-flux boundary conditions for y-direction for vnew
    do iy = 1, LLy
        vnew(0, iy) = vnew(2, iy)
        vnew(LLx+1, iy) = vnew(LLx-1, iy)
    end do
    
    ! Second update step
    do ix = 1, LLx
        do iy = 1, LLy
            v(ix, iy) = vnew(ix, iy) + slmbdax * (vnew(ix+1, iy) + vnew(ix-1, iy) - 2.0d0 * vnew(ix, iy)) &
                                      + slmbday * (vnew(ix, iy+1) + vnew(ix, iy-1) - 2.0d0 * vnew(ix, iy))
        end do
    end do
    
    return
end subroutine Euler_forward

subroutine Euler_forward_optimized(LLx, LLy, v, vnew, slmbdax, slmbday)
    implicit none
    integer, intent(in) :: LLx, LLy
    real(kind=8), intent(inout) :: v(0:LLx+1, 0:LLy+1), vnew(0:LLx+1, 0:LLy+1)
    real(kind=8), intent(in) :: slmbdax, slmbday
    integer :: ix, iy
    
    ! Set boundary conditions first
    ! Set non-flux boundary conditions for y-direction
    !$OMP PARALLEL DO
    do iy = 1, LLy
        v(0, iy) = v(2, iy)
        v(LLx+1, iy) = v(LLx-1, iy)
    end do
    !$OMP END PARALLEL DO
    
    ! Set non-flux boundary conditions for x-direction
    !$OMP PARALLEL DO
    do ix = 1, LLx
        v(ix, 0) = v(ix, 2)
        v(ix, LLy+1) = v(ix, LLy-1)
    end do
    !$OMP END PARALLEL DO
    
    ! Corner boundary conditions
    v(0, 0) = v(2, 2)
    v(0, LLy+1) = v(2, LLy-1)
    v(LLx+1, 0) = v(LLx-1, 2)
    v(LLx+1, LLy+1) = v(LLx-1, LLy-1)
    
    ! First update step - can be parallelized
    !$OMP PARALLEL DO PRIVATE(ix)
    do iy = 1, LLy
        do ix = 1, LLx
            vnew(ix, iy) = v(ix, iy) + slmbdax * (v(ix+1, iy) + v(ix-1, iy) - 2.0d0 * v(ix, iy)) &
                                     + slmbday * (v(ix, iy+1) + v(ix, iy-1) - 2.0d0 * v(ix, iy))
        end do
    end do
    !$OMP END PARALLEL DO
    
    ! Update boundary conditions for vnew
    !$OMP PARALLEL DO
    do iy = 1, LLy
        vnew(0, iy) = vnew(2, iy)
        vnew(LLx+1, iy) = vnew(LLx-1, iy)
    end do
    !$OMP END PARALLEL DO
    
    !$OMP PARALLEL DO
    do ix = 1, LLx
        vnew(ix, 0) = vnew(ix, 2)
        vnew(ix, LLy+1) = vnew(ix, LLy-1)
    end do
    !$OMP END PARALLEL DO
    
    ! Corner boundary conditions for vnew
    vnew(0, 0) = vnew(2, 2)
    vnew(0, LLy+1) = vnew(2, LLy-1)
    vnew(LLx+1, 0) = vnew(LLx-1, 2)
    vnew(LLx+1, LLy+1) = vnew(LLx-1, LLy-1)
    
    ! Second update step - can also be parallelized
    !$OMP PARALLEL DO PRIVATE(ix)
    do iy = 1, LLy
        do ix = 1, LLx
            v(ix, iy) = vnew(ix, iy) + slmbdax * (vnew(ix+1, iy) + vnew(ix-1, iy) - 2.0d0 * vnew(ix, iy)) &
                                      + slmbday * (vnew(ix, iy+1) + vnew(ix, iy-1) - 2.0d0 * vnew(ix, iy))
        end do
    end do
    !$OMP END PARALLEL DO
    
    return
end subroutine Euler_forward_optimized

! Maybe implement other methods such as RK4 