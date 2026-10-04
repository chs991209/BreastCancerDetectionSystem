"""
[Natural EMBED — 자연 유병률(통계적 안정 구성)]
EMBED 라벨파일(EMBED_OpenData_metadata/clinical_reduced.csv) 기준 2D 전체 자연 비율
= 17,594 : 10,113 → abnormal 36.5%. 디스크의 균형 서브셋(7000:7000)에서 이 자연 비율을
'무작위 표집'으로 재현한다(고정 시드 → 모든 method/seed 가 동일한 자연 test 를 봄).

설계 원칙(논문 설명 가능·객관):
- **random-sampled / naturally-picked** — crc32 같은 결정적 '조건화(conditioning)' 배열이 아니라
  고정 시드 RandomState 로 무작위 추출 → 특정 배열에 과적합('too-conditioned')되지 않음.
- **statistically stabilized** — 자연 test 집합을 모든 실험에서 동일하게 고정 → method 비교가
  같은 test 위에서 이뤄져 DeLong 가능(‘method 관점의 통계적 불균형’ 없음).
- 라벨/환자split 은 부모(EMBEDDataset)에서 그대로 계승(누수 없음). 음성 전량 유지 + 양성 무작위 표집.
전처리 캐시는 균형 EMBED('embed')와 공유(같은 uid).
"""
import os
import numpy as np
from typing import List

from transfer_learning.embed.dataset import EMBEDDataset

_NAT_RATE = 0.365           # EMBED 라벨파일 전체 2D 자연 abnormal 비율
_SPLIT_SALT = {'train': 101, 'val': 202, 'test': 303}   # split별 고정(문자열 hash 랜덤화 회피)
_BASE_SEED = 20260825       # 고정 → 모든 run 에서 동일한 무작위 표본(재현·안정)


class NaturalEMBEDDataset(EMBEDDataset):
    name = 'embed_nat'

    def _build_records(self) -> List[dict]:
        recs = super()._build_records()            # 라벨·환자split·on-disk 필터(부모, 객관)
        neg = [x for x in recs if x['label'] == 0]
        pos = [x for x in recs if x['label'] == 1]
        keep = min(len(pos), int(round(len(neg) * _NAT_RATE / (1.0 - _NAT_RATE))))
        rs = np.random.RandomState(_BASE_SEED + _SPLIT_SALT[self.split])  # 고정 시드 무작위
        idx = rs.permutation(len(pos))[:keep]
        out = neg + [pos[i] for i in idx]
        print(f"[{self.name}:{self.split}] natural {100*_NAT_RATE:.1f}% (random, fixed seed): "
              f"neg={len(neg)} + pos {keep}/{len(pos)} → n={len(out)} "
              f"({100*keep/len(out):.1f}% abnormal)")
        return out

    def _load_or_build_index(self, rebuild: bool):
        # 자연 표본은 부모 balanced 인덱스와 다르므로 항상 새로 구성(무작위 표집은 고정 시드라 재현).
        return self._build_records()

    def _preproc_cache_path(self, rec: dict) -> str:
        return os.path.join(self.cache_dir, 'embed', 'preproc', self.split, f"{rec['uid']}.npz")
