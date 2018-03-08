# Copyright (c) 2015-2017 Michael Eastwood
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

function tikhonov(transfermatrix, mmodes; regularization=1e-2)
    lmax = mmax = transfermatrix.lmax
    alm  = Alm(lmax, mmax)

    pool  = CachingPool(workers())
    queue = reverse(collect(0:mmax))

    lck = ReentrantLock()
    prg = Progress(length(queue))
    increment() = (lock(lck); next!(prg); unlock(lck))

    @sync for worker in workers()
        @async while length(queue) > 0
            m = pop!(queue)
            _alm = remotecall_fetch(_tikhonov, pool, transfermatrix, mmodes,
                                    regularization, lmax, m)
            for l = m:lmax
                alm[l, m] = _alm[l-m+1]
            end
            increment()
        end
    end
    alm
end

function _tikhonov(transfermatrix, mmodes, regularization, lmax, m)
    BLAS.set_num_threads(16)
    BB = zeros(Complex128, lmax-m+1, lmax-m+1)
    Bv = zeros(Complex128, lmax-m+1)
    permutation = baseline_permutation(transfermatrix, m)
    for β = 1:length(frequencies(mmodes))
        _tikhonov_accumulate!(BB, Bv, transfermatrix[m, β], mmodes[m, β], permutation)
    end
    _tikhonov_inversion(BB, Bv, regularization)
end

function _tikhonov_accumulate!(BB, Bv, B, v, permutation)
    v = v[permutation]
    f = v .== 0 # flags
    v = v[.!f]
    B = B[.!f, :]
    B′ = B'
    BB .+= B′*B
    Bv .+= B′*v
end

function _tikhonov_inversion(BB, Bv, regularization)
    (BB + regularization*I) \ Bv
end

