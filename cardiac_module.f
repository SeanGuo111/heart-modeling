      subroutine cardiac_simulation(Lx, Ly, nstim, iseed, rbcl, Dfu, 
     &                   gicai, gtos, gtof, gnacai, zxr, 
     &                   nbt, cxinit, xnai,
     &                   xnao, xki, xko, cao, temp, xxr, xf, dt,
     &                   v_out, cb_out, csrb_out, ci_out, t_out, num_steps)
          
          !f2py intent(in) Lx, Ly, nstim, iseed, rbcl, Dfu
          !f2py intent(in) gicai, gtos, gtof, gnacai, zxr
          !f2py intent(in) nbt, cxinit, xnai
          !f2py intent(in) xnao, xki, xko, cao, temp, xxr, xf, dt
          !f2py intent(out) v_out, cb_out, csrb_out, ci_out, t_out
          !f2py intent(out) num_steps
          
          implicit double precision (a-h,o-z)
          
          ! Define array dimensions based on input parameters
          double precision v(0:Lx+1,0:Ly+1), vnew(0:Lx+1,0:Ly+1)
          double precision dv(0:Lx,0:Ly), vold(0:Lx,0:Ly)
          double precision xm(Lx,Ly), xh(Lx,Ly), xj(Lx,Ly), xr(Lx,Ly)
          double precision xs1(Lx,Ly), qks(Lx,Ly), xkur(Lx,Ly), ykur(Lx,Ly)
          double precision xtof(Lx,Ly), ytof(Lx,Ly)
          double precision xtos(Lx,Ly), ytos(Lx,Ly)
          
          double precision cb(Lx,Ly), ci(Lx,Ly), csrb(Lx,Ly), csri(Lx,Ly)
          double precision cnsr(Lx,Ly), po(Lx,Ly), c1(Lx,Ly), c2(Lx,Ly)
          double precision xi1(Lx,Ly), xi2(Lx,Ly), xi2s(Lx,Ly), c1s(Lx,Ly)
          double precision c2s(Lx,Ly), xi1s(Lx,Ly), cit(Lx,Ly), cbt(Lx,Ly)
          
          ! Output arrays
          double precision v_out(Lx,Ly,nstim*int(rbcl/dt)/100+1)
          double precision cb_out(Lx,Ly,nstim*int(rbcl/dt)/100+1)
          double precision csrb_out(Lx,Ly,nstim*int(rbcl/dt)/100+1)
          double precision ci_out(Lx,Ly,nstim*int(rbcl/dt)/100+1)
          double precision t_out(nstim*int(rbcl/dt)/100+1)
          
          ! Other arrays from the original code
          double precision xicaqz(Lx,Ly), xinacaqz(Lx,Ly)
          double precision xitor(Lx,Ly), xitorf(Lx,Ly), zina(Lx,Ly)
          double precision zica(Lx,Ly)
          double precision pi(Lx,Ly), pb(Lx,Ly), pox(Lx,Ly), pos(Lx,Ly)
          double precision ra(Lx,Ly)
          integer nsb(Lx,Ly), nsi(Lx,Ly)
          double precision gicaz(0:Lx,0:Ly), gnacaz(0:Lx,0:Ly)        
          double precision pbxz(0:Lx,0:Ly)
          
          ! Calculate derived parameters
          frt = xf/(xxr*temp)
          dx = 0.015d0
          dy = 0.015d0
          duration = rbcl
          nstep = duration/dt
          slmbdax = Dfu*dt/4.0d0/dx/dx
          slmbday = Dfu*dt/4.0d0/dy/dy
          
          ! Initialize output counter
          output_count = 1
          
          ! Initial conditions (same as original code)
          do ix = 1, Lx
              do iy = 1, Ly
            
	         ! Ca variables
	             cb(ix,iy)=0.1d0  ! boundary cytosolic Ca
	             ci(ix,iy)=0.1d0  ! interior cytosolic Ca
         
                 csrb(ix,iy)=cxinit  ! boundary SR Ca
	             csri(ix,iy)=cxinit  ! interior SR Ca
	            	
                 po(ix,iy)=0.0d0  ! LCC model facing cb 
	             c1(ix,iy)=0.0d0  
                 c2(ix,iy)=1.0d0
                 xi1(ix,iy)=0.0d0
                 xi2(ix,iy)=0.0d0
	         
	             pos(ix,iy)=0.0d0  ! LCC model facing spark
	             c1s(ix,iy)=0.0d0
                 c2s(ix,iy)=0.0d0
                 xi1s(ix,iy)=0.0d0
	             xi2s(ix,iy)=0.0d0
         
	            call total(ci(ix,iy),cit(ix,iy))
	            call total(cb(ix,iy),cbt(ix,iy))
         
	            nsb(ix,iy)=5
	            v(ix,iy)=-90.0d0
	            xm(ix,iy)=0.001d0 ! Ina gate variable
  	            xh(ix,iy)=1.0d0   ! Ina gate variable
	            xj(ix,iy)=1.0d0   ! Ina gate variable
                xr(ix,iy)=0.0d0
                xs1(ix,iy)=0.3d0 ! iks gate var. 
	            qks(ix,iy)=0.2d0 ! Ca gating of iks
                xtos(ix,iy)=0.01d0 ! ito activation
                ytos(ix,iy)=0.9d0 ! ito inactivation
	            xtof(ix,iy)=0.02d0 ! ito fast activation
                ytof(ix,iy)=0.8d0  ! ito slow inactivation
                vold(ix,iy) = v(ix,iy)
	            gicaz(ix,iy)=gicai
	            gnacaz(ix,iy)=gnacai
	            pbxz(ix,iy)=pbxi
              enddo
          enddo
          
          ! Integration loop (similar to original)
          t = 0.0
          
          ! Store initial state
          do ix = 1, Lx
              do iy = 1, Ly
                  v_out(ix,iy,output_count) = v(ix,iy)
                  cb_out(ix,iy,output_count) = cb(ix,iy)
                  csrb_out(ix,iy,output_count) = csrb(ix,iy)
                  ci_out(ix,iy,output_count) = ci(ix,iy)
              enddo
          enddo
          t_out(output_count) = t
          output_count = output_count + 1
          
          do iz = 1, nstim  ! number of beats
              nstep = int(rbcl/dt)
              
              do ncount = 0, nstep
                  ! Main simulation code (copied from original)
                  
                  ! Loop over all grid points
                  do iy = 1, Ly
                      do ix = 1, Lx
                      gica=gicaz(ix,iy)
                      	gnaca=gnacaz(ix,iy)
                      	pbx=pbxz(ix,iy)

c ********************************************************************

	                    call xfree(cit(ix,iy),ci(ix,iy))  ! convert total Ca to free
	                    call xfree(cbt(ix,iy),cb(ix,iy))  ! convert total Ca to free

c ******* fraction of clusters with sparks ***************************

	                     pb(ix,iy)=dfloat(nsb(ix,iy))/dfloat(nbt)

c*********************************************************************

                         vupb=0.4
	                     vupi=0.4
                     
                             call uptake(cb(ix,iy),vupb,xupb)  ! uptake at boundary 
	                     call uptake(ci(ix,iy),vupi,xupi)  ! uptake at interior sites

c *******************************************************************************

	                     call inaca(v(ix,iy),frt,xnai,xnao,cao,cb(ix,iy),xinacaq1)
                     
	                     xinacaq=gnaca*xinacaq1
	                     
	                     xinacaqz(ix,iy)=xinacaq
                     
	                     pox(ix,iy)=po(ix,iy)+pos(ix,iy)
	                 call ica(v(ix,iy),frt,cao,cb(ix,iy),pox(ix,iy),rca,xicaq) ! LCC current
                         
	                     xicaq=gica*130.0*xicaq

c ******************************************************************************* 	
                     
	                     ! spark rate at junctional dyads
	                     
	                     qq=0.5d0
	                     ab=35.0*qq
	                     csrx=600.0d0
	                     phisr=1.0/(1.0+(csrx/csrb(ix,iy))**10) ! cutoff rate bellow 500 muM
                     
	                     alphab=ab*dabs(rca)*po(ix,iy)*phisr ! spark on rate due to LCC 
	                     bts=1.0/30.0   ! spark off rate 
                     
	                     call markov(dt,v(ix,iy),cb(ix,iy),c1(ix,iy),
                          +  c2(ix,iy),xi1(ix,iy),xi2(ix,iy),po(ix,iy),
                          +  c1s(ix,iy),c2s(ix,iy),
                          +  xi1s(ix,iy), xi2s(ix,iy),pos(ix,iy),alphab,bts,zxr)
                     
c *****************************************************************************

	                     gsrb=(0.01/1.5)*1.0
	                     xryrb=gsrb*csrb(ix,iy)*pb(ix,iy)    ! ryr current for boundary region

c ******************************************************************************
	                     ! spark rate in the interior
	                     
	                     xryri=0.0d0

c *****************************************************************************	

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


c ********* time evolution due to binomial distribution *****************

	                    nsbx=nsb(ix,iy)
	                    call binevol(nbt,nsbx,alphab,bts,dt,iseed,ndeltapx,ndeltamx)
                    
	                    if(ndeltamx.gt.nsbx.or.ndeltapx.gt.nbt) then
	                    nsb(ix,iy)=0
	                    else
	                    nsb(ix,iy)=nsb(ix,iy)+ndeltapx-ndeltamx
	                    endif 


c  ********** do voltage here with adaptive time step ********************
	
	 ! pass currents from Ca system to voltage 

	                   wca=12.0d0
                           xinaca=wca*xinacaq  ! convert ion flow to current
                           xica=2.0d0*wca*xicaq
                           
                           zica(ix,iy)=xica

c -------------  time step adjustment ------------------------
      
                      adq=dabs(dv(ix,iy))
                      
                      if(adq.gt.25.0d0) then ! finer time step when dv/dt large
                      mstp=10
                      else
                      mstp=1
                      endif 
	                    
		      hode=dt/dfloat(mstp) 
      
                      do iii=1 , mstp  
      
c************ these are the voltage dependent currents ***********************
    
	                call ina(hode,v(ix,iy),frt,xh(ix,iy),xj(ix,iy),
                     &    xm(ix,iy),xnai,xnao,xina) ! sodium
                          
	                zina(ix,iy)=xina
                
                
	                call ikr(hode,v(ix,iy),frt,xko,xki,
                     &    xr(ix,iy),xikr) ! ikr
                
                      	call iks(hode,v(ix,iy),frt,cb(ix,iy),xnao,xnai,xko,xki,
                     &    xs1(ix,iy),qks(ix,iy),xiks) ! iks
                
                        call ik1(hode,v(ix,iy),frt,xki,xko,xik1)  ! ik1
                
                      	call ito(hode,v(ix,iy),frt,xki,xko,xtof(ix,iy),
                     &  ytof(ix,iy),xtos(ix,iy),ytos(ix,iy),xito,gtof,gtos) ! ito
                
                        call inak(v(ix,iy),frt,xko,xnao,xnai,xinak) ! inak
                
	                xitor(ix,iy)=xito

c**********UNIFORM STIMULATION *************************************
          
	!if(time.lt.1.0) then
	!stim=80.0d0
	!else
	!stim=0.0
	!endif       
c       EDGE STIMULATION 

        if(time.lt.1.0) then 
          if(ix.lt.10 .and. iy.lt.10) then ! stim corrner
             stim = 80.0d0
          else
             stim = 0.0
          endif
          else
             stim = 0.0
          endif

c*************************************************************************
	
        dvh=-(xina+xik1+xikr+xiks+xito+xinaca+xica+xinak)+ stim

	

        v(ix,iy)=v(ix,iy)+dvh*hode

	         enddo 
                      enddo
                  enddo
                  
                  ! Apply diffusion with Euler Forward
                  call Euler_Forward(Lx, Ly, v, vnew, slmbdax, slmbday)
                  call Euler_Forward(Lx, Ly, v, vnew, slmbdax, slmbday)
                  
                  ! Store output at desired intervals
                  if (mod(ncount, 100) .eq. 0) then
                      do ix = 1, Lx
                          do iy = 1, Ly
                              v_out(ix,iy,output_count) = v(ix,iy)
                              cb_out(ix,iy,output_count) = cb(ix,iy)
                              csrb_out(ix,iy,output_count) = csrb(ix,iy)
                              ci_out(ix,iy,output_count) = ci(ix,iy)
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
      end subroutine cardiac_simulation
      


c ****************************************************************************
      
      subroutine Euler_forward(LLx,LLy,v,vnew,slmbdax,slmbday)
      implicit double precision (a-h,o-z)
        
      double precision  v(0:LLx+1,0:LLy+1),vnew(0:LLx+1,0:LLy+1)
        
         v(0,0)=v(2,2)
         v(0,LLy+1)=v(2,LLy-1)
         v(LLx+1,0)=v(LLx-1,2)
         v(LLx+1,LLy+1)=v(LLx-1,LLy-1)

         do ix=1,LLx ! non flux bc
           v(ix,0)=v(ix,2)
           v(ix,LLy+1)=v(ix,LLy-1)
         enddo
  
         do iy=1,LLy
           v(0,iy)=v(2,iy)
           v(LLx+1,iy)=v(LLx-1,iy)
          enddo
 
         do ix=1,LLx
         do iy=1,LLy
        vnew(ix,iy)=v(ix,iy)+slmbdax*(v(ix+1,iy)+v(ix-1,iy)
     #  -2.0d0*v(ix,iy))
     #     +slmbday*(v(ix,iy+1)+v(ix,iy-1)-2.0d0*v(ix,iy))
         enddo
         enddo
 

         vnew(0,0)=vnew(2,2)
         vnew(0,LLy+1)=vnew(2,LLy-1)
         vnew(LLx+1,0)=vnew(LLx-1,2)
         vnew(LLx+1,LLy+1)=vnew(LLx-1,LLy-1)

         do ix=1,LLx
           vnew(ix,0)=vnew(ix,2)
           vnew(ix,LLy+1)=vnew(ix,LLy-1)
         enddo

          do iy=1,LLy
           vnew(0,iy)=vnew(2,iy)
           vnew(LLx+1,iy)=vnew(LLx-1,iy)
          enddo

         do ix=1,LLx
         do iy=1,LLy
         v(ix,iy)=vnew(ix,iy)+slmbdax*(vnew(ix+1,iy)+vnew(ix-1,iy)
     #   -2.0d0*vnew(ix,iy))
     #     +slmbday*(vnew(ix,iy+1)+vnew(ix,iy-1)-2.0d0*vnew(ix,iy))
         enddo
         enddo
          return

          end