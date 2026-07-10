//! Reusable packed factorizations preserving the caller's storage ownership.

use num_traits::Zero;
use crate::{backend::{PositiveDefinitePackedBackend, SymmetricPackedBackend, HermitianPackedBackend}, storage::{PackedStorage,PackedStorageMut}, PackedMatrixError};

pub(crate) fn checked_n(n:usize)->Result<i32,PackedMatrixError>{i32::try_from(n).map_err(|_|PackedMatrixError::DimensionOverflow{n})}
pub(crate) fn check_rhs<T>(n:usize,rhs:&[T])->Result<(),PackedMatrixError>{if rhs.len()==n{Ok(())}else{Err(PackedMatrixError::InvalidVectorLength{expected:n,actual:rhs.len()})}}
pub(crate) fn check_rhs_many<T>(n:usize,nrhs:usize,rhs:&[T])->Result<(),PackedMatrixError>{let expected=n.checked_mul(nrhs).ok_or(PackedMatrixError::DimensionOverflow{n})?;if rhs.len()==expected{Ok(())}else{Err(PackedMatrixError::InvalidVectorLength{expected,actual:rhs.len()})}}
pub(crate) fn check_info(info:i32,message:&'static str)->Result<(),PackedMatrixError>{if info<0{Err(PackedMatrixError::LapackIllegalArgument{argument:-info})}else if info>0{Err(PackedMatrixError::FactorizationFailure{index:info as usize,message})}else{Ok(())}}

#[derive(Clone,Debug)]
pub struct PackedCholesky<T,S=Vec<T>>{pub(crate)n:usize,pub(crate)data:S,pub(crate)uplo:u8,pub(crate)marker:std::marker::PhantomData<T>}
pub type PackedCholeskyViewMut<'a,T>=PackedCholesky<T,&'a mut[T]>;
impl<T,S> PackedCholesky<T,S> where T:PositiveDefinitePackedBackend,S:PackedStorageMut<T>{
    pub(crate) fn factorize_storage(n:usize,mut data:S,uplo:u8)->Result<Self,PackedMatrixError>{let mut info=0;unsafe{T::pptrf(uplo,checked_n(n)?,data.as_mut_slice(),&mut info)};check_info(info,"matrix is not positive definite")?;Ok(Self{n,data,uplo,marker:std::marker::PhantomData})}
    pub fn inverse_in_place(&mut self)->Result<(),PackedMatrixError>{let mut info=0;unsafe{T::pptri(self.uplo,checked_n(self.n)?,self.data.as_mut_slice(),&mut info)};check_info(info,"packed Cholesky inverse failed")}
}
impl<T,S> PackedCholesky<T,S> where T:PositiveDefinitePackedBackend,S:PackedStorage<T>{
    pub fn dimension(&self)->usize{self.n} pub fn as_slice(&self)->&[T]{self.data.as_slice()}
    pub fn solve_vector_in_place(&self,rhs:&mut[T])->Result<(),PackedMatrixError>{self.solve_many_in_place(rhs,1)}
    pub fn solve_many_in_place(&self,rhs:&mut[T],nrhs:usize)->Result<(),PackedMatrixError>{check_rhs_many(self.n,nrhs,rhs)?;let n=checked_n(self.n)?;let mut info=0;unsafe{T::pptrs(self.uplo,n,checked_n(nrhs)?,self.as_slice(),rhs,n,&mut info)};check_info(info,"packed Cholesky solve failed")}
    pub fn solve_vector(&self,rhs:&[T])->Result<Vec<T>,PackedMatrixError>{let mut out=rhs.to_vec();self.solve_vector_in_place(&mut out)?;Ok(out)}
}
impl<T> PackedCholesky<T,Vec<T>>{pub fn into_vec(self)->Vec<T>{self.data}}

#[derive(Clone,Debug)]
pub struct PackedSymmetricFactor<T,S=Vec<T>>{pub(crate)n:usize,pub(crate)data:S,pub(crate)pivots:Vec<i32>,pub(crate)uplo:u8,pub(crate)marker:std::marker::PhantomData<T>}
pub type PackedSymmetricFactorViewMut<'a,T>=PackedSymmetricFactor<T,&'a mut[T]>;
impl<T,S> PackedSymmetricFactor<T,S> where T:SymmetricPackedBackend,S:PackedStorageMut<T>{
    pub(crate) fn factorize_storage(n:usize,mut data:S,uplo:u8)->Result<Self,PackedMatrixError>{let mut pivots=vec![0;n];let mut info=0;unsafe{T::sptrf(uplo,checked_n(n)?,data.as_mut_slice(),&mut pivots,&mut info)};check_info(info,"symmetric packed matrix is singular")?;Ok(Self{n,data,pivots,uplo,marker:std::marker::PhantomData})}
    pub fn inverse_in_place(&mut self)->Result<(),PackedMatrixError>{let mut work=vec![T::zero();self.n];let mut info=0;unsafe{T::sptri(self.uplo,checked_n(self.n)?,self.data.as_mut_slice(),&self.pivots,&mut work,&mut info)};check_info(info,"symmetric packed inverse failed")}
}
impl<T,S> PackedSymmetricFactor<T,S> where T:SymmetricPackedBackend,S:PackedStorage<T>{
    pub fn dimension(&self)->usize{self.n} pub fn as_slice(&self)->&[T]{self.data.as_slice()} pub fn pivots(&self)->&[i32]{&self.pivots}
    pub fn solve_vector_in_place(&self,rhs:&mut[T])->Result<(),PackedMatrixError>{self.solve_many_in_place(rhs,1)}
    pub fn solve_many_in_place(&self,rhs:&mut[T],nrhs:usize)->Result<(),PackedMatrixError>{check_rhs_many(self.n,nrhs,rhs)?;let n=checked_n(self.n)?;let mut info=0;unsafe{T::sptrs(self.uplo,n,checked_n(nrhs)?,self.as_slice(),&self.pivots,rhs,n,&mut info)};check_info(info,"symmetric packed solve failed")}
    pub fn solve_vector(&self,rhs:&[T])->Result<Vec<T>,PackedMatrixError>{let mut out=rhs.to_vec();self.solve_vector_in_place(&mut out)?;Ok(out)}
}
impl<T> PackedSymmetricFactor<T,Vec<T>>{pub fn into_vec(self)->Vec<T>{self.data}}

#[derive(Clone,Debug)]
pub struct PackedHermitianFactor<T,S=Vec<T>>{pub(crate)n:usize,pub(crate)data:S,pub(crate)pivots:Vec<i32>,pub(crate)uplo:u8,pub(crate)marker:std::marker::PhantomData<T>}
pub type PackedHermitianFactorViewMut<'a,T>=PackedHermitianFactor<T,&'a mut[T]>;
impl<T,S> PackedHermitianFactor<T,S> where T:HermitianPackedBackend,S:PackedStorageMut<T>{
    pub(crate) fn factorize_storage(n:usize,mut data:S,uplo:u8)->Result<Self,PackedMatrixError>{let mut pivots=vec![0;n];let mut info=0;unsafe{T::hptrf(uplo,checked_n(n)?,data.as_mut_slice(),&mut pivots,&mut info)};check_info(info,"Hermitian packed matrix is singular")?;Ok(Self{n,data,pivots,uplo,marker:std::marker::PhantomData})}
    pub fn inverse_in_place(&mut self)->Result<(),PackedMatrixError>{let mut work=vec![T::zero();self.n];let mut info=0;unsafe{T::hptri(self.uplo,checked_n(self.n)?,self.data.as_mut_slice(),&self.pivots,&mut work,&mut info)};check_info(info,"Hermitian packed inverse failed")}
}
impl<T,S> PackedHermitianFactor<T,S> where T:HermitianPackedBackend,S:PackedStorage<T>{
    pub fn dimension(&self)->usize{self.n} pub fn as_slice(&self)->&[T]{self.data.as_slice()} pub fn pivots(&self)->&[i32]{&self.pivots}
    pub fn solve_vector_in_place(&self,rhs:&mut[T])->Result<(),PackedMatrixError>{self.solve_many_in_place(rhs,1)}
    pub fn solve_many_in_place(&self,rhs:&mut[T],nrhs:usize)->Result<(),PackedMatrixError>{check_rhs_many(self.n,nrhs,rhs)?;let n=checked_n(self.n)?;let mut info=0;unsafe{T::hptrs(self.uplo,n,checked_n(nrhs)?,self.as_slice(),&self.pivots,rhs,n,&mut info)};check_info(info,"Hermitian packed solve failed")}
    pub fn solve_vector(&self,rhs:&[T])->Result<Vec<T>,PackedMatrixError>{let mut out=rhs.to_vec();self.solve_vector_in_place(&mut out)?;Ok(out)}
}
impl<T> PackedHermitianFactor<T,Vec<T>>{pub fn into_vec(self)->Vec<T>{self.data}}
