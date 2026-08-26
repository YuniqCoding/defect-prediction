# -*- coding: utf-8 -*-
import streamlit as st

# 올린 CSV 파일을 표로 읽으려고 가져온다
import pandas as pd

# 학습용과 시험용으로 나누려고 가져온다
from sklearn.model_selection import train_test_split

# 고를 수 있는 모델 세 가지를 가져온다
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 로지스틱 회귀 앞에 값 크기를 맞추는 단계를 붙이려고 가져온다
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 채점에 쓸 네 가지 점수를 가져온다
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# 그림을 그리려고 가져온다 - 화면 없는 곳에서도 그려지도록 Agg 방식으로 맞춘다
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 그림을 파일로 저장할 자리를 잡으려고 가져온다
from pathlib import Path

# notes.md 에서 번호로 시작하는 줄을 찾으려고 가져온다
import re

# .env 에 넣어둔 열쇠를 읽고, 글 만들기 서비스를 부르려고 가져온다
import os
import requests
from dotenv import load_dotenv

# PDF 를 서버에 저장하지 않고 메모리 위에서 만들려고 가져온다
from io import BytesIO

# PDF 를 만들려고 가져온다
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)

# 지금 시각을 읽어오려고 가져온다
from datetime import datetime

# 그래프 안 한글이 네모로 깨지지 않게, 이 컴퓨터에 있는 한글 글꼴을 먼저 지정한다
설치된_글꼴 = {글꼴.name for 글꼴 in font_manager.fontManager.ttflist}
한글_글꼴 = None
for 후보 in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim", "Dotum"]:
    if 후보 in 설치된_글꼴:
        plt.rcParams["font.family"] = 후보
        한글_글꼴 = 후보
        break

# 글꼴을 바꾸면 음수 부호가 네모로 깨지므로 그 처리도 같이 꺼준다
plt.rcParams["axes.unicode_minus"] = False

# 그림을 저장할 폴더 - app.py 옆에 figures 폴더를 만들어 둔다
그림_폴더 = Path(__file__).parent / "figures"
그림_폴더.mkdir(exist_ok=True)

# 요약 다섯 줄이 적힌 글 파일 - app.py 옆의 notes.md 를 그대로 읽어 쓴다
노트_경로 = Path(__file__).parent / "notes.md"

# 열쇠는 코드 파일에 적지 않는다 - 내 컴퓨터에서는 secom-project 맨 위의 .env 에서 읽는다
열쇠_파일 = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(열쇠_파일)

# 열쇠 이름은 두 자리에서 같은 것을 쓴다
열쇠_이름 = "GOOGLE_API_KEY"


def 열쇠_찾기():
    """열쇠를 두 군데에서 찾는다. (열쇠, 어디서_찾았나) 를 돌려주고, 없으면 (None, None).

    - 내 컴퓨터 : .env 의 GOOGLE_API_KEY
    - 올린 자리 : 배포 서비스의 비밀 값(secrets) 에 있는 같은 이름
    열쇠 값은 코드 파일에 적지 않는다.
    """
    # 1) 내 컴퓨터 - .env 를 읽어 환경변수로 올려둔 것을 먼저 본다
    열쇠 = os.getenv(열쇠_이름)
    if 열쇠 and 열쇠.strip():
        return 열쇠.strip(), ".env 파일"

    # 2) 올린 자리 - 배포 서비스의 비밀 값
    #    비밀 값 파일이 아예 없으면 st.secrets 를 건드리는 것만으로 탈이 나므로 감싼다
    try:
        열쇠 = st.secrets[열쇠_이름]
    except Exception:
        return None, None

    if 열쇠 and str(열쇠).strip():
        return str(열쇠).strip(), "배포 서비스의 비밀 값"
    return None, None

# 문장을 만들어 줄 서비스 자리와 모델 이름
글_만들기_주소 = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "gemini-3.7-flash:generateContent")


def 부탁_글_짓기(자료):
    """지금 화면의 숫자만 담은 부탁 글을 만든다 - 여기 없는 숫자는 넘어가지 않는다."""
    내_줄 = 자료["점수_표"].iloc[0]        # 첫 줄이 내 모델
    기준_줄 = 자료["점수_표"].iloc[1]      # 둘째 줄이 기준 모델
    return f"""당신은 공정 데이터 분석 결과를 현장 담당자에게 설명하는 사람입니다.
아래 숫자만 보고 한국어로 해석 문장을 써 주세요.

[지금 화면의 숫자]
- 문턱(임계값): {자료['문턱']:.2f}
- 기준 모델(전부 정상이라 답함) 점수: 정확도 {기준_줄['정확도']}, 정밀도 {기준_줄['정밀도']}, 재현율 {기준_줄['재현율']}, F1 {기준_줄['F1']}
- 내 모델({자료['모델_이름']}) 점수: 정확도 {내_줄['정확도']}, 정밀도 {내_줄['정밀도']}, 재현율 {내_줄['재현율']}, F1 {내_줄['F1']}
- 지목한 건수: {자료['지목_건수']}건
- 그중 진짜 불량: {자료['진짜_건수']}건
- 놓친 불량: {자료['놓친_건수']}건

[지켜야 할 것]
- 세 문장을 넘기지 마세요.
- 위에 적힌 숫자만 쓰세요. 새로운 숫자를 만들거나 계산해서 넣지 마세요.
- 무엇이 원인이라고 단정하지 마세요. 놓친 건을 말할 때는 "~한 구간에 몰려 있었다" 정도로만 쓰세요.
- 좋다 나쁘다 하는 평가 대신, 무엇이 어떠했는지만 담담하게 쓰세요.
- 문장만 쓰고 제목이나 머리말은 붙이지 마세요."""


@st.cache_resource
def 피디에프_글꼴_준비():
    """한글이 네모로 깨지지 않게 글꼴 파일을 PDF 안에 심을 수 있도록 등록한다."""
    if 한글_글꼴 is None:
        return None, None
    try:
        # 위에서 고른 한글 글꼴의 실제 파일 자리를 찾아온다
        보통_길 = Path(font_manager.findfont(한글_글꼴, fallback_to_default=False))
    except Exception:
        return None, None

    pdfmetrics.registerFont(TTFont("한글", str(보통_길)))
    굵은_이름 = "한글"

    # 굵은 글꼴 파일이 옆에 같이 있으면 그것도 등록한다 (malgun.ttf 옆의 malgunbd.ttf)
    굵은_길 = 보통_길.with_name(보통_길.stem + "bd" + 보통_길.suffix)
    if 굵은_길.exists():
        pdfmetrics.registerFont(TTFont("한글굵게", str(굵은_길)))
        굵은_이름 = "한글굵게"
    return "한글", 굵은_이름


def 리포트_피디에프_만들기(자료, 다섯줄, 해석_문장, 오늘):
    """지금 화면에 뜬 값만 담아 PDF 를 만들어 바이트로 돌려준다 - 파일로 저장하지 않는다."""
    보통, 굵게 = 피디에프_글꼴_준비()
    if 보통 is None:
        return None

    제목틀 = ParagraphStyle("제목틀", fontName=굵게, fontSize=18, leading=25, spaceAfter=3)
    날짜틀 = ParagraphStyle("날짜틀", fontName=보통, fontSize=10, leading=15,
                          textColor=colors.HexColor("#555555"), spaceAfter=10)
    소제목 = ParagraphStyle("소제목", fontName=굵게, fontSize=13, leading=18,
                         spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor("#1F3B63"))
    줄머리 = ParagraphStyle("줄머리", fontName=굵게, fontSize=10, leading=15,
                         spaceBefore=7, spaceAfter=1)
    본문틀 = ParagraphStyle("본문틀", fontName=보통, fontSize=9.5, leading=15, spaceAfter=3)

    # 메모리 위에 만든다 - 서버 디스크에 파일을 남기지 않는다
    통 = BytesIO()
    문서 = SimpleDocTemplate(통, pagesize=A4,
                           leftMargin=20*mm, rightMargin=20*mm,
                           topMargin=18*mm, bottomMargin=18*mm,
                           title="설비 측정값으로 고장을 미리 알아채기")

    내용 = [Paragraph("설비 측정값으로 고장을 미리 알아채기", 제목틀),
           Paragraph(오늘, 날짜틀),
           HRFlowable(width="100%", thickness=1,
                      color=colors.HexColor("#1F3B63"), spaceAfter=8)]

    # 1) 화면에 지금 떠 있는 요약 다섯 줄
    내용.append(Paragraph("프로젝트 요약 다섯 줄", 소제목))
    if 다섯줄:
        for 머리, 몸 in 다섯줄:
            내용.append(Paragraph(머리, 줄머리))
            내용.append(Paragraph(몸, 본문틀))
    else:
        내용.append(Paragraph("화면에 요약 다섯 줄이 없습니다", 본문틀))

    # 2) 점수 표 - 화면의 표를 그대로 옮긴다
    내용.append(Paragraph("결과 표", 소제목))
    표 = 자료["점수_표"]
    칸들 = [[Paragraph(f"<b>{이름}</b>", 본문틀) for 이름 in 표.columns]]
    for _, 줄 in 표.iterrows():
        칸들.append([Paragraph(str(줄[이름]), 본문틀) for 이름 in 표.columns])
    표그림 = Table(칸들, colWidths=[62*mm] + [26*mm] * (len(표.columns) - 1))
    표그림.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF4")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAAAAA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    내용.append(표그림)

    # 3) 지금 손잡이 값에서 나온 건수
    내용.append(Paragraph("지금 문턱에서의 건수", 소제목))
    내용.append(Paragraph(
        f"문턱 {자료['문턱']:.2f} · 모델 {자료['모델_이름']} · "
        f"적은 쪽 가중치 {'켬' if 자료['가중치_켬'] else '끔'}", 본문틀))
    내용.append(Paragraph(
        f"시험용 {자료['시험_행수']}행, 그중 진짜 불량 {자료['전체_불량']}건 — "
        f"지목한 건수 {자료['지목_건수']}건, 그중 진짜 불량 {자료['진짜_건수']}건, "
        f"놓친 불량 {자료['놓친_건수']}건", 본문틀))

    # 4) 맨 아래 - 방금 받은 해석 문장
    내용.append(Paragraph("해석 문장", 소제목))
    if 해석_문장:
        내용.append(Paragraph(해석_문장, 본문틀))
    else:
        내용.append(Paragraph("아직 만들지 않았습니다", 본문틀))

    내용.append(Spacer(1, 10))
    내용.append(Paragraph(
        "이 리포트의 숫자는 화면에 떠 있던 값을 그대로 옮긴 것입니다.", 날짜틀))

    문서.build(내용)
    return 통.getvalue()


# 실패했을 때 화면에 내보낼 말 - 빨간 글씨 대신 이 한 줄만 보여준다
못_만듦_말 = "지금은 문장을 만들 수 없습니다. 다시 눌러주세요"
열쇠_문제_말 = ("열쇠가 없거나 잘못된 것 같습니다. "
            ".env 파일의 GOOGLE_API_KEY 를 확인한 뒤 다시 눌러주세요")

# 두 자리 어디에도 열쇠가 없을 때 - 이 한 줄만 보여주고 앱은 그대로 둔다
열쇠_없음_말 = "열쇠가 없습니다"


def 숫자_도장(자료):
    """지금 화면 숫자를 글자 한 줄로 묶는다 - 같은 숫자인지 견주는 데 쓴다."""
    내_줄 = 자료["점수_표"].iloc[0]
    기준_줄 = 자료["점수_표"].iloc[1]
    조각 = [f"{자료['문턱']:.2f}", str(자료["모델_이름"]), str(자료["가중치_켬"]),
           str(자료["지목_건수"]), str(자료["진짜_건수"]), str(자료["놓친_건수"])]
    # 점수 표의 여덟 칸까지 넣어야, 다시 학습해서 점수만 바뀐 것도 다른 숫자로 본다
    조각 += [str(줄[칸]) for 줄 in (내_줄, 기준_줄)
            for 칸 in ("정확도", "정밀도", "재현율", "F1")]
    return "|".join(조각)


def 해석_문장_받기(자료, 열쇠):
    """부탁 글을 보내고 문장을 받아온다. (문장, 알릴_말, 열쇠_문제인가) 를 돌려준다."""
    try:
        응답 = requests.post(
            글_만들기_주소,
            # 열쇠는 머리말에만 실어 보낸다 - 화면에도, 주소에도 남기지 않는다
            headers={"x-goog-api-key": 열쇠, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": 부탁_글_짓기(자료)}]}],
                  "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}},
            timeout=60)
    except UnicodeError:
        # .env 의 열쇠에 한글 같은 글자가 섞여 있으면 보내기 전에 여기서 걸린다
        return None, 열쇠_문제_말, True
    except requests.RequestException:
        # 인터넷이 끊겼거나 시간이 오래 걸린 경우 - 자세한 사정은 화면에 늘어놓지 않는다
        return None, 못_만듦_말, False
    except Exception:
        # 그 밖에 무슨 일이 나더라도 앱이 빨간 오류로 죽지 않게 한 줄로 알린다
        return None, 못_만듦_말, False

    if 응답.status_code != 200:
        # 열쇠가 잘못되면 401, 403 이 오고, 400 과 함께 열쇠 이야기가 오기도 한다
        열쇠_문제 = 응답.status_code in (401, 403)
        if 응답.status_code == 400 and "API_KEY" in 응답.text.upper():
            열쇠_문제 = True
        # 응답 내용에 열쇠가 섞여 있을 수 있으므로 그 내용은 화면에 내보내지 않는다
        return None, (열쇠_문제_말 if 열쇠_문제 else 못_만듦_말), 열쇠_문제

    후보들 = 응답.json().get("candidates", [])
    if not 후보들:
        return None, 못_만듦_말, False
    조각들 = 후보들[0].get("content", {}).get("parts", [])
    문장 = "".join(조각.get("text", "") for 조각 in 조각들).strip()
    if not 문장:
        return None, 못_만듦_말, False
    return 문장, None, False


def 표로_읽기(올린_파일):
    """올린 CSV 를 표로 읽는다. 못 읽으면 (None, 알릴_말) 을 돌려준다."""
    # 엑셀에서 저장한 한글 CSV 는 cp949 인 경우가 많아 두 글자표로 차례로 해본다
    for 글자표 in ("utf-8", "cp949"):
        올린_파일.seek(0)          # 앞서 읽다 만 자리를 처음으로 되돌린다
        try:
            return pd.read_csv(올린_파일, encoding=글자표), None
        except UnicodeDecodeError:
            continue               # 글자가 깨지면 다음 글자표로 한 번 더 해본다
        except pd.errors.EmptyDataError:
            return None, "파일에 읽을 내용이 없습니다. 다른 파일을 올려주세요"
        except pd.errors.ParserError:
            return None, ("칸 수가 줄마다 달라 표로 읽지 못했습니다. "
                          "파일을 열어 확인한 뒤 다시 올려주세요")
        except Exception:
            # 그 밖에 무슨 일이 나더라도 빨간 오류로 죽지 않게 한 줄로 알린다
            return None, "이 파일은 표로 읽지 못했습니다. 다른 파일을 올려주세요"
    return None, ("글자가 깨져 읽지 못했습니다. "
                  "UTF-8 이나 CP949 로 저장한 CSV 를 올려주세요")


def 다섯줄_읽기():
    """notes.md 의 '리포트 뼈대 다섯 줄' 대목을 (제목, 내용) 짝으로 뽑아온다."""
    # 파일이 없거나 그 대목이 아직 없으면 빈 목록을 돌려준다 - 지어내지 않는다
    if not 노트_경로.exists():
        return []
    글 = 노트_경로.read_text(encoding="utf-8")
    표시 = "### 리포트 뼈대 다섯 줄"
    if 표시 not in 글:
        return []

    # 그 표시 다음부터 다음 소제목(###) 앞까지가 다섯 줄이 적힌 대목이다
    조각 = 글.split(표시)[1].split("###")[0]

    줄들, 제목, 몸통 = [], None, []
    for 한줄 in 조각.strip().splitlines():
        한줄 = 한줄.strip()
        if not 한줄:
            continue
        # '1)' 이든 '1.' 이든 번호로 시작하면 새 줄의 제목이다
        if re.match(r"^\d+[).]\s", 한줄):
            if 제목:
                줄들.append((제목, " ".join(몸통)))
            제목, 몸통 = 한줄, []
        else:
            몸통.append(한줄)
    if 제목:
        줄들.append((제목, " ".join(몸통)))
    return 줄들

제목 = "설비 측정값으로 고장을 미리 알아채기"

# 브라우저 탭에 뜨는 이름도 같은 제목으로
st.set_page_config(page_title=제목)

# 화면 맨 위 큰 제목과 그 아래 작은 글씨
st.title(제목)
st.caption("공정 조건 다섯 가지로 고장 여부를 판별합니다")

# Streamlit 1.59.0 은 화면을 다시 그릴 때 안 고른 탭 내용까지 밑에 같이 보여주는 문제가 있다
# 안 고른 탭에는 data-inert 표시가 붙으므로, 그 칸만 감추는 규칙을 넣어 임시로 막는다
st.markdown(
    """<style>[data-testid="stTabs"] div[data-inert="true"]{display:none !important;}</style>""",
    unsafe_allow_html=True,
)

# 탭 다섯 개 - 앞의 두 개만 채우고 나머지는 아직 비워둔다
탭_이름 = ["데이터 훑기", "전처리", "학습", "결과", "리포트"]
탭들 = st.tabs(탭_이름)

# 탭 1 - 데이터 훑기
with 탭들[0]:
    # CSV 파일을 끌어다 놓는 자리
    올린_파일 = st.file_uploader("CSV 파일을 올려주세요", type="csv")

    if 올린_파일 is None:
        # 아직 파일이 없으면 안내 한 줄만 보여준다
        st.write("파일을 올려주세요")
        # 파일을 뺐으면 다음 탭이 옛 자료를 쓰지 않도록 보관해 둔 것도 지운다
        st.session_state.pop("데이터", None)
        st.session_state.pop("결과_열", None)
    else:
        # 올린 파일을 표로 읽어들인다 - 못 읽는 파일이면 빨간 오류 대신 한 줄로 알린다
        데이터, 못읽은_까닭 = 표로_읽기(올린_파일)

        if 데이터 is None:
            st.warning(못읽은_까닭)
            # 못 읽었으면 다음 탭이 옛 자료를 쓰지 않도록 보관해 둔 것도 지운다
            st.session_state.pop("데이터", None)
            st.session_state.pop("결과_열", None)
        else:

            # 다음 탭에서도 다시 올리지 않고 그대로 쓰도록 보관해 둔다
            st.session_state["데이터"] = 데이터

            # 1) 행 수와 열 수를 한 줄로
            st.write(f"행 {len(데이터)}개, 열 {len(데이터.columns)}개")

            # 2) 앞의 다섯 줄을 표로
            st.write("앞의 다섯 줄")
            st.dataframe(데이터.head())

            # 3) 열마다 빈칸이 몇 개인지 세고, 빈칸이 하나라도 있는 열만 남긴다
            빈칸_개수 = 데이터.isna().sum()
            빈칸_개수 = 빈칸_개수[빈칸_개수 > 0]

            st.write("빈칸이 있는 열")
            if len(빈칸_개수) == 0:
                # 빈칸이 아예 없으면 표 대신 그 사실을 알려준다
                st.write("빈칸이 있는 열이 없습니다")
            else:
                # 빈칸이 많은 열이 위로 오게 내림차순으로 정렬한다
                빈칸_개수 = 빈칸_개수.sort_values(ascending=False)
                # 열 이름 / 빈칸 개수 / 빈칸 비율 세 칸짜리 표를 만든다
                빈칸_표 = pd.DataFrame({
                    "열 이름": 빈칸_개수.index,
                    "빈칸 개수": 빈칸_개수.values,
                    # 비율은 전체 행 수로 나눠 백분율로 바꾸고 소수 첫째 자리까지만 남긴다
                    "빈칸 비율(%)": (빈칸_개수.values / len(데이터) * 100).round(1),
                })
                st.dataframe(빈칸_표, hide_index=True)

            # 4) 결과 열을 고르는 선택 상자
            고른_열 = st.selectbox("결과 열을 고르세요", 데이터.columns)

            # 어느 열을 결과로 골랐는지도 다음 탭에서 쓰도록 보관해 둔다
            st.session_state["결과_열"] = 고른_열

            # 고른 열의 값별 개수를 센다 - 빈칸도 하나의 값으로 같이 센다
            값별_개수 = 데이터[고른_열].value_counts(dropna=False)
            값별_표 = pd.DataFrame({
                "값": 값별_개수.index.astype(str),
                "개수": 값별_개수.values,
            })
            st.write(f"'{고른_열}' 열의 값별 개수")
            st.dataframe(값별_표, hide_index=True)

# 탭 2 - 전처리
with 탭들[1]:
    if "데이터" not in st.session_state:
        # 첫 탭에서 올린 파일을 그대로 쓰므로, 아직 없으면 그것만 알려준다
        st.write("먼저 '데이터 훑기' 탭에서 CSV 파일을 올려주세요")
    else:
        # 첫 탭에서 올린 파일을 다시 올리지 않고 그대로 가져다 쓴다
        데이터 = st.session_state["데이터"]
        결과_열 = st.session_state["결과_열"]

        st.write(f"'데이터 훑기'에서 올린 파일을 그대로 씁니다 - 결과 열 : '{결과_열}'")

        # 표 전체에 빈칸이 몇 개인지 먼저 센다
        빈칸_전 = int(데이터.isna().sum().sum())
        st.write(f"지금 표 전체의 빈칸 : {빈칸_전}개")

        if 빈칸_전 == 0:
            # 채울 것이 없으면 선택 상자를 띄우지 않는다
            st.write("빈칸이 없습니다. 채울 것이 없어요")
            채우기_방법 = "없음"
        else:
            채우기_방법 = st.selectbox("빈칸을 무엇으로 채울까요", ["중앙값", "평균", "0"])

        # 글자로 된 열을 찾는다 - 결과 열은 따로 다루므로 뺀다
        # pandas 3 부터 글자 열 종류가 object 가 아니라 str 로 나오므로, 숫자가 아닌 열을 글자 열로 본다
        글자_열 = [열 for 열 in 데이터.columns
                 if 열 != 결과_열 and not pd.api.types.is_numeric_dtype(데이터[열])]

        if len(글자_열) == 0:
            st.write("글자로 된 열이 없습니다")
            글자_처리 = "없음"
        else:
            st.write("글자로 된 열 : " + ", ".join(글자_열))
            글자_처리 = st.radio("이 열들을 어떻게 할까요",
                              ["학습에서 빼기", "숫자로 바꾸기"], horizontal=True)

        # 결과 열에 실제로 들어 있는 값 중에서 1로 볼 값을 고른다
        결과_값들 = [str(값) for 값 in 데이터[결과_열].dropna().unique()]
        일로_볼_값 = st.selectbox(f"'{결과_열}' 열의 어느 값을 1로 볼까요", 결과_값들)

        # 학습용을 몇 퍼센트로 할지 고른다 - 기본은 80퍼센트, 즉 8대 2
        학습_퍼센트 = st.slider("학습용 비율(%)", 50, 90, 80, 5)
        st.write(f"학습용 {학습_퍼센트 // 10} : 시험용 {(100 - 학습_퍼센트) // 10} 으로 나눕니다")

        if st.button("적용"):
            # 원본은 건드리지 않고 복사본으로 작업한다
            작업 = 데이터.copy()

            # 1) 글자 열 처리
            if 글자_처리 == "학습에서 빼기":
                작업 = 작업.drop(columns=글자_열)
                글자_설명 = f"글자 열 {len(글자_열)}개를 학습에서 뺐습니다 - {', '.join(글자_열)}"
            elif 글자_처리 == "숫자로 바꾸기":
                for 열 in 글자_열:
                    # 같은 글자끼리 같은 번호를 준다 - 빈칸은 -1 이 된다
                    작업[열] = pd.factorize(작업[열])[0]
                글자_설명 = f"글자 열 {len(글자_열)}개를 숫자로 바꿨습니다 - {', '.join(글자_열)}"
            else:
                글자_설명 = "글자로 된 열이 없어 손댈 것이 없었습니다"

            # 2) 결과 열을 0과 1로 바꾼다 - 고른 값이면 1, 아니면 0
            정답 = (작업[결과_열].astype(str) == 일로_볼_값).astype(int)
            입력 = 작업.drop(columns=[결과_열])

            # 바꾼 뒤 1이 몇 건인지 세어 확인한다
            일_개수 = int(정답.sum())
            st.write(f"결과 열을 0과 1로 바꿨습니다 - 1은 {일_개수}건, 0은 {len(정답) - 일_개수}건")

            if 일_개수 == 0:
                # 1이 한 건도 없으면 고른 값이 그 열에 없다는 뜻이므로 여기서 멈춘다
                st.error(f"1이 0건입니다. '{일로_볼_값}' 값이 '{결과_열}' 열에 없어 더 진행할 수 없습니다")
                st.stop()

            # 3) 남은 빈칸을 고른 방법으로 채운다
            if 채우기_방법 == "중앙값":
                입력 = 입력.fillna(입력.median(numeric_only=True))
            elif 채우기_방법 == "평균":
                입력 = 입력.fillna(입력.mean(numeric_only=True))
            elif 채우기_방법 == "0":
                입력 = 입력.fillna(0)

            # 채우고 난 뒤 학습에 쓸 표에 빈칸이 몇 개 남았는지 다시 센다
            빈칸_후 = int(입력.isna().sum().sum())

            # 0개가 안 됐으면 어느 열이 남았는지, 왜 안 채워졌는지 알려준다
            if 빈칸_후 > 0:
                남은_열 = 입력.isna().sum()
                남은_열 = 남은_열[남은_열 > 0].sort_values(ascending=False)
                # 열이 통째로 비어 있으면 중앙값도 평균도 구할 수 없어 못 채운다
                남은_표 = pd.DataFrame({
                    "열 이름": 남은_열.index,
                    "남은 빈칸": 남은_열.values,
                    "안 채워진 까닭": ["열이 통째로 비어 있어 채울 값을 구할 수 없음"
                                 if int(입력[열].isna().sum()) == len(입력)
                                 else "숫자가 아닌 열이라 채우는 값을 구할 수 없음"
                                 for 열 in 남은_열.index],
                })
                st.warning(f"채우고도 빈칸이 {빈칸_후}개 남았습니다")
                st.dataframe(남은_표, hide_index=True)

            # 4) 학습용과 시험용으로 나눈다
            시험_비율 = (100 - 학습_퍼센트) / 100
            try:
                # 1과 0의 비율이 양쪽에 고르게 들어가도록 나눈다
                학습_입력, 시험_입력, 학습_정답, 시험_정답 = train_test_split(
                    입력, 정답, test_size=시험_비율, random_state=42, stratify=정답)
                나눔_설명 = "1과 0의 비율을 맞춰서 나눴습니다"
            except ValueError:
                # 한쪽 값이 너무 적어 비율을 맞출 수 없으면 그냥 나눈다
                학습_입력, 시험_입력, 학습_정답, 시험_정답 = train_test_split(
                    입력, 정답, test_size=시험_비율, random_state=42)
                나눔_설명 = "한쪽 값이 너무 적어 비율을 맞추지 않고 나눴습니다"

            # 나눈 두 조각의 행 수를 더해 원본 행 수와 같은지 확인한다 - 행이 사라지지 않았는지 보는 것
            합친_행수 = len(학습_입력) + len(시험_입력)
            원본_행수 = len(데이터)

            # 다음 탭(학습)에서 쓰도록 나눈 결과를 보관해 둔다
            st.session_state["나눈_자료"] = (학습_입력, 시험_입력, 학습_정답, 시험_정답)

            st.write("---")

            # 1) 빈칸이 몇 개에서 몇 개로 줄었는지
            st.write(f"빈칸 : 표 전체 {빈칸_전}개 → 학습에 쓸 표 {빈칸_후}개")

            # 2) 글자 열을 어떻게 처리했는지
            st.write(글자_설명)

            # 3) 학습용과 시험용 행 수
            st.write(f"학습용 {len(학습_입력)}행, 시험용 {len(시험_입력)}행 - {나눔_설명}")

            # 4) 양쪽의 1 개수와 비율을 표로
            나눔_표 = pd.DataFrame({
                "구분": ["학습용", "시험용"],
                "행 수": [len(학습_정답), len(시험_정답)],
                "1 개수": [int(학습_정답.sum()), int(시험_정답.sum())],
                "1 비율(%)": [round(float(학습_정답.mean()) * 100, 1),
                            round(float(시험_정답.mean()) * 100, 1)],
            })
            st.dataframe(나눔_표, hide_index=True)

            # 5) 행 수가 원본과 같은지 확인한 결과를 알려준다
            if 합친_행수 == 원본_행수:
                st.write(f"행 수 확인 : 원본 {원본_행수}행 = 학습용 + 시험용 {합친_행수}행 - 사라진 행 없음")
            else:
                st.error(f"행 수 확인 : 원본 {원본_행수}행인데 나눈 뒤 {합친_행수}행 - "
                         f"{원본_행수 - 합친_행수}행이 맞지 않습니다")

# 탭 3 - 학습
with 탭들[2]:
    if "나눈_자료" not in st.session_state:
        # 두 번째 탭에서 '적용'을 아직 누르지 않았으면 이 한 줄만 보여준다
        st.write("전처리를 먼저 해주세요")
    else:
        # 전처리 탭에서 나눠둔 네 조각을 그대로 가져다 쓴다
        학습_입력, 시험_입력, 학습_정답, 시험_정답 = st.session_state["나눈_자료"]

        st.write(f"'전처리'에서 나눈 자료를 그대로 씁니다 - "
                 f"학습용 {len(학습_입력)}행, 시험용 {len(시험_입력)}행")

        # 어떤 모델로 학습할지 고르는 선택 상자
        모델_이름 = st.selectbox("모델을 고르세요",
                              ["로지스틱 회귀", "의사결정나무", "랜덤 포레스트"])

        # 적은 쪽(불량)을 더 무겁게 쳐서 학습시킬지 켜고 끄는 스위치
        가중치_켬 = st.toggle("적은 쪽에 가중치 주기")

        if st.button("학습"):
            # 빈칸이 하나라도 남아 있으면 모델이 학습하지 못하므로 여기서 멈추고 까닭을 알려준다
            남은_빈칸 = int(학습_입력.isna().sum().sum()) + int(시험_입력.isna().sum().sum())
            if 남은_빈칸 > 0:
                st.error(f"빈칸이 {남은_빈칸}개 남아 있어 학습할 수 없습니다 - "
                         f"'전처리' 탭에서 빈칸을 채운 뒤 다시 해주세요")
                st.stop()

            # 학습이 아예 안 되는 두 경우를 먼저 걸러낸다
            # - 그냥 두면 sklearn 이 빨간 오류를 내며 앱이 멈춘다
            못하는_까닭 = None
            if 학습_정답.nunique() < 2:
                못하는_까닭 = ("학습용 자료의 정답이 0과 1 중 한 가지뿐이라 학습할 수 없습니다 "
                          "- '전처리' 탭에서 결과 열이나 1로 볼 값을 다시 골라주세요")
            elif 학습_입력.shape[1] == 0:
                못하는_까닭 = ("학습에 쓸 열이 하나도 없습니다 "
                          "- '전처리' 탭에서 글자 열을 '숫자로 바꾸기' 로 골라보세요")

            if 못하는_까닭:
                st.warning(못하는_까닭)
            else:
                # 스위치가 켜져 있으면 적은 쪽에 가중치를 주는 설정을 넣는다
                가중치 = "balanced" if 가중치_켬 else None

                # 고른 이름에 맞는 모델을 만든다 - 다시 눌러도 같은 결과가 나오게 씨앗을 42로 고정한다
                if 모델_이름 == "로지스틱 회귀":
                    # 열마다 값의 크기가 크게 다르면 학습이 잘 안 되므로 크기를 먼저 맞춘 뒤 학습한다
                    모델 = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight=가중치, random_state=42),
                    )
                elif 모델_이름 == "의사결정나무":
                    모델 = DecisionTreeClassifier(class_weight=가중치, random_state=42)
                else:
                    모델 = RandomForestClassifier(n_estimators=100, class_weight=가중치,
                                                random_state=42)

                # 학습용으로 학습시킨다 - 오래 걸릴 수 있어 도는 중이라는 표시를 띄운다
                with st.spinner("학습하는 중입니다"):
                    모델.fit(학습_입력, 학습_정답)

                # 시험용으로 채점한다 - 학습에 쓰지 않은 자료로 맞혀보는 것
                예측 = 모델.predict(시험_입력)

                # 불량일 가능성도 여기서 딱 한 번만 구해 둔다
                # - '결과' 탭에서 문턱을 옮길 때 이 값을 다시 자르기만 하면 되므로 다시 학습할 일이 없다
                가능성 = 모델.predict_proba(시험_입력)[:, 1]

                # 비교할 기준 모델 - 아무것도 배우지 않고 전부 정상(0)이라고만 답한다
                기준_예측 = [0] * len(시험_정답)

                # 네 가지 점수를 한 번에 구한다 - 불량이라 지목한 게 하나도 없으면 0으로 둔다
                def 점수내기(정답, 예측값):
                    return [
                        round(float(accuracy_score(정답, 예측값)), 3),
                        round(float(precision_score(정답, 예측값, zero_division=0)), 3),
                        round(float(recall_score(정답, 예측값, zero_division=0)), 3),
                        round(float(f1_score(정답, 예측값, zero_division=0)), 3),
                    ]

                내_점수 = 점수내기(시험_정답, 예측)
                기준_점수 = 점수내기(시험_정답, 기준_예측)

                # 고른 모델과 기준 모델의 점수를 한 표에 나란히 놓는다
                점수_표 = pd.DataFrame({
                    "구분": [모델_이름, "전부 정상이라 답하는 기준 모델"],
                    "정확도": [내_점수[0], 기준_점수[0]],
                    "정밀도": [내_점수[1], 기준_점수[1]],
                    "재현율": [내_점수[2], 기준_점수[2]],
                    "F1": [내_점수[3], 기준_점수[3]],
                })
                st.dataframe(점수_표, hide_index=True)

                # 불량을 몇 건이나 잡아냈는지 숫자로도 한 줄 남긴다
                잡은_건수 = int(((예측 == 1) & (시험_정답.values == 1)).sum())
                st.write(f"시험용 불량 {int(시험_정답.sum())}건 중 {잡은_건수}건을 잡아냈습니다 - "
                         f"불량이라 지목한 건 모두 {int((예측 == 1).sum())}건")

                # 다음 탭(결과)에서 쓰도록 학습한 모델과 채점 결과를 보관해 둔다
                st.session_state["학습_결과"] = {
                    "모델": 모델,
                    "모델_이름": 모델_이름,
                    "가중치_켬": 가중치_켬,
                    "예측": 예측,
                    # 채점할 때 쓴 시험용 정답도 같이 넣어둔다 - '결과' 탭이 이것만 보고 그리게 한다
                    "시험_정답": 시험_정답,
                    # 문턱을 옮길 때 다시 자르기만 하면 되도록 가능성을 통째로 넣어둔다
                    "가능성": 가능성,
                    # 중요 변수 그림에 쓸 열 이름도 같이 넣어둔다
                    "열_이름": list(시험_입력.columns),
                    "점수_표": 점수_표,
                }

# 탭 4 - 결과
with 탭들[3]:
    if "학습_결과" not in st.session_state:
        # 세 번째 탭에서 '학습'을 아직 누르지 않았으면 이 한 줄만 보여준다
        st.write("학습을 먼저 해주세요")
    else:
        # 학습 탭이 넘겨준 꾸러미를 그대로 꺼내 쓴다 - 여기서 다시 학습하지 않는다
        학습_결과 = st.session_state["학습_결과"]
        모델_이름 = 학습_결과["모델_이름"]
        시험_정답 = 학습_결과["시험_정답"]

        # 어떤 설정으로 학습한 결과를 보고 있는지 한 줄로 알려준다
        가중치_글 = "켬" if 학습_결과["가중치_켬"] else "끔"
        st.write(f"'학습' 탭의 결과를 그대로 씁니다 - 모델 '{모델_이름}', "
                 f"적은 쪽 가중치 {가중치_글}, 시험용 {len(시험_정답)}행")

        # 예전 방식으로 학습해 가능성이 없는 꾸러미면 여기서 멈추고 다시 학습을 권한다
        if "가능성" not in 학습_결과:
            st.warning("예전 방식으로 학습된 결과입니다 - '학습' 탭에서 학습을 한 번 더 눌러주세요")
            st.stop()

        # 1) 문턱 슬라이더 - 가능성이 이 값을 넘으면 불량이라고 본다
        문턱 = st.slider("문턱(임계값)", 0.05, 0.95, 0.50, 0.05)
        st.write(f"지금 문턱 : {문턱:.2f} - 불량일 가능성이 {문턱:.2f} 이상이면 불량이라고 봅니다")

        # 학습할 때 한 번 구해둔 가능성을 문턱에서 다시 자르기만 한다
        # - 여기에는 fit 도 predict 도 없다. 그래서 슬라이더를 옮겨도 다시 학습하지 않는다
        가능성 = 학습_결과["가능성"]
        예측 = (가능성 >= 문턱).astype(int)

        # 슬라이더를 옮길 때마다 아래 숫자가 전부 이 예측에서 다시 나온다
        정답_값 = 시험_정답.values
        지목_건수 = int((예측 == 1).sum())
        진짜_건수 = int(((예측 == 1) & (정답_값 == 1)).sum())
        놓친_건수 = int(((예측 == 0) & (정답_값 == 1)).sum())

        칸1, 칸2, 칸3 = st.columns(3)
        칸1.metric("지목한 건수", f"{지목_건수}건")
        칸2.metric("그중 진짜 불량", f"{진짜_건수}건")
        칸3.metric("놓친 불량", f"{놓친_건수}건")

        # 문턱을 옮기면 이 세 점수도 같이 움직인다 - 지목한 게 없으면 0으로 둔다
        내_정밀도 = round(float(precision_score(시험_정답, 예측, zero_division=0)), 3)
        내_재현율 = round(float(recall_score(시험_정답, 예측, zero_division=0)), 3)
        내_f1 = round(float(f1_score(시험_정답, 예측, zero_division=0)), 3)

        칸4, 칸5, 칸6 = st.columns(3)
        칸4.metric("정밀도", f"{내_정밀도:.3f}")
        칸5.metric("재현율", f"{내_재현율:.3f}")
        칸6.metric("F1", f"{내_f1:.3f}")

        # 2) 기준 모델과 내 모델을 나란히 놓은 점수 표 - 이것도 지금 문턱으로 다시 매긴다
        기준_예측 = [0] * len(시험_정답)
        점수_표 = pd.DataFrame({
            "구분": [f"{모델_이름} (문턱 {문턱:.2f})", "전부 정상이라 답하는 기준 모델"],
            "정확도": [round(float(accuracy_score(시험_정답, 예측)), 3),
                     round(float(accuracy_score(시험_정답, 기준_예측)), 3)],
            "정밀도": [내_정밀도,
                     round(float(precision_score(시험_정답, 기준_예측, zero_division=0)), 3)],
            "재현율": [내_재현율,
                     round(float(recall_score(시험_정답, 기준_예측, zero_division=0)), 3)],
            "F1": [내_f1,
                   round(float(f1_score(시험_정답, 기준_예측, zero_division=0)), 3)],
        })
        st.write("점수 견주기")
        st.dataframe(점수_표, hide_index=True)

        # 리포트 탭이 문턱 슬라이더를 그대로 따라오도록, 지금 문턱의 결과를 넘겨준다
        # - 리포트 탭은 이 꾸러미만 보고 그리므로 거기서 다시 셈하지 않는다
        st.session_state["리포트_자료"] = {
            "문턱": 문턱,
            "모델_이름": 모델_이름,
            "가중치_켬": 학습_결과["가중치_켬"],
            "점수_표": 점수_표,
            "지목_건수": 지목_건수,
            "진짜_건수": 진짜_건수,
            "놓친_건수": 놓친_건수,
            "시험_행수": len(시험_정답),
            "전체_불량": int(시험_정답.sum()),
        }

        # 3) 그 아래 - 혼동행렬을 네 칸으로 풀어서 보여준다
        #    labels=[0, 1] 로 자리를 고정해야 한쪽 값이 없어도 네 칸이 그대로 나온다
        정상을_정상, 헛경보, 놓친_것, 잡은_것 = confusion_matrix(
            시험_정답, 예측, labels=[0, 1]).ravel()

        st.write("혼동행렬 - 시험용을 네 칸으로 나눈 것")
        혼동_표 = pd.DataFrame({
            "칸": ["잡은 것", "놓친 것", "헛경보", "정상을 정상이라 한 것"],
            "건수": [int(잡은_것), int(놓친_것), int(헛경보), int(정상을_정상)],
            "무슨 뜻인가": [
                "불량을 불량이라 맞힌 것 - 많을수록 좋다",
                "불량인데 정상이라고 넘긴 것 - 가장 아픈 실수다",
                "정상인데 불량이라 지목한 것 - 괜히 세워 보는 헛걸음이다",
                "정상을 정상이라 맞힌 것 - 그냥 지나간 것이다",
            ],
        })
        st.dataframe(혼동_표, hide_index=True)

        # 네 칸을 더하면 시험용 행 수와 같은지 확인한다 - 빠진 건이 없는지 보는 것
        네_칸_합 = int(잡은_것) + int(놓친_것) + int(헛경보) + int(정상을_정상)
        if 네_칸_합 == len(시험_정답):
            st.write(f"칸 수 확인 : 네 칸을 더하면 {네_칸_합}건 = 시험용 {len(시험_정답)}행 - 빠진 건 없음")
        else:
            st.error(f"칸 수 확인 : 네 칸 합이 {네_칸_합}건인데 시험용은 {len(시험_정답)}행 - 맞지 않습니다")

        # 4) 그 아래 - 문턱별 비교 표 (0.1 부터 0.9 까지 아홉 줄)
        #    여기도 학습하지 않는다 - 학습 때 구해둔 가능성을 문턱마다 다시 자르기만 한다
        st.write("---")
        st.write("문턱별 비교")

        비교_줄 = []
        for 후보_문턱 in [칸 / 10 for 칸 in range(1, 10)]:
            후보_예측 = (가능성 >= 후보_문턱).astype(int)
            비교_줄.append({
                "문턱": f"{후보_문턱:.1f}",
                "지목 건수": int((후보_예측 == 1).sum()),
                "그중 진짜": int(((후보_예측 == 1) & (정답_값 == 1)).sum()),
                "놓친 건수": int(((후보_예측 == 0) & (정답_값 == 1)).sum()),
                "정밀도": round(float(precision_score(시험_정답, 후보_예측, zero_division=0)), 3),
                "재현율": round(float(recall_score(시험_정답, 후보_예측, zero_division=0)), 3),
                "F1": round(float(f1_score(시험_정답, 후보_예측, zero_division=0)), 3),
            })

        비교_표 = pd.DataFrame(비교_줄)
        최고_F1 = float(비교_표["F1"].max())

        if 최고_F1 == 0:
            # 어느 문턱에서도 불량을 한 건도 못 잡았으면 표시할 줄이 없다
            비교_표["표시"] = ""
            최고_글 = "어느 문턱에서도 F1이 0입니다 - 불량을 한 건도 잡지 못했습니다"
        else:
            # F1 이 가장 높은 줄에 표시를 붙인다 - 같은 값이 여럿이면 그 줄에 모두 붙인다
            비교_표["표시"] = ["◀ F1 가장 높음" if float(값) == 최고_F1 else ""
                            for 값 in 비교_표["F1"]]
            최고_문턱들 = [줄["문턱"] for 줄 in 비교_줄 if float(줄["F1"]) == 최고_F1]
            최고_글 = (f"F1이 가장 높은 문턱은 {', '.join(최고_문턱들)} 이고, "
                     f"그때 F1은 {최고_F1:.3f} 입니다")

        st.dataframe(비교_표, hide_index=True)
        st.write(최고_글)

        # 5) 그 아래 - 그림 세 장
        st.write("---")
        st.write("그림 세 장")

        # 한글 글꼴을 못 찾았으면 글자가 깨질 수 있다는 것을 먼저 알려준다
        if 한글_글꼴 is None:
            st.warning("한글 글꼴을 찾지 못했습니다 - 그래프 글자가 네모로 깨질 수 있습니다")
        else:
            st.caption(f"그래프 글꼴 : {한글_글꼴} / 저장 자리 : {그림_폴더}")

        열_이름 = 학습_결과["열_이름"]

        # --- 그림 1) 중요 변수 - 어느 항목이 판단에 많이 쓰였는지 ---
        if 모델_이름 == "로지스틱 회귀":
            # 파이프라인 안의 로지스틱 회귀에서 계수를 꺼내고, 부호는 떼고 크기만 본다
            중요도 = abs(학습_결과["모델"].named_steps["logisticregression"].coef_[0])
            중요도_뜻 = "계수의 크기 - 클수록 그 항목이 판단을 크게 흔들었다는 뜻입니다"
        else:
            # 나무 계열은 갈라질 때 얼마나 도움이 됐는지를 그대로 알려준다
            중요도 = 학습_결과["모델"].feature_importances_
            중요도_뜻 = "나무가 갈라질 때 쓰인 정도 - 클수록 자주, 크게 쓰였다는 뜻입니다"

        # 중요도가 큰 항목이 위로 오게 정렬하고, 열이 많을 때를 대비해 위에서 15개만 그린다
        중요_표 = pd.DataFrame({"항목": 열_이름, "중요도": 중요도})
        중요_표 = 중요_표.sort_values("중요도", ascending=False)
        보여줄_개수 = min(15, len(중요_표))
        보여줄 = 중요_표.head(보여줄_개수).iloc[::-1]      # 가로 막대는 아래부터 그려지므로 뒤집는다

        그림1, 축1 = plt.subplots(figsize=(7, max(3, 보여줄_개수 * 0.35)))
        축1.barh(보여줄["항목"].astype(str), 보여줄["중요도"], color="#4C78A8")
        축1.set_xlabel("중요도")
        축1.set_title(f"중요 변수 - {모델_이름}")
        그림1.tight_layout()
        경로1 = 그림_폴더 / "01_중요변수.png"
        그림1.savefig(경로1, dpi=120)
        st.pyplot(그림1)
        plt.close(그림1)                                  # 그린 뒤 닫아야 그림이 쌓이지 않는다
        st.caption(f"어느 항목이 판단에 많이 쓰였는지 - {중요도_뜻}. "
                   f"전체 {len(중요_표)}개 중 위에서 {보여줄_개수}개만 그렸습니다 (저장 : {경로1.name})")

        # --- 그림 2) 혼동행렬 그림 ---
        행렬 = confusion_matrix(시험_정답, 예측, labels=[0, 1])

        그림2, 축2 = plt.subplots(figsize=(6, 5))
        칠한것 = 축2.imshow(행렬, cmap="Blues")
        축2.set_xticks([0, 1], ["정상이라 함", "불량이라 함"])
        축2.set_yticks([0, 1], ["진짜 정상", "진짜 불량"])
        축2.set_xlabel("모델이 내놓은 답")
        축2.set_ylabel("실제 값")
        축2.set_title(f"혼동행렬 (문턱 {문턱:.2f})")

        # 네 칸 안에 이름과 건수를 직접 적어 넣는다
        칸_이름 = [["정상을 정상이라 함", "헛경보"], ["놓친 것", "잡은 것"]]
        for 세로 in range(2):
            for 가로 in range(2):
                값 = int(행렬[세로][가로])
                # 칸 색이 진하면 검은 글씨가 안 보이므로 흰 글씨로 바꾼다
                글자색 = "white" if 값 > 행렬.max() / 2 else "black"
                # 칸 이름과 건수를 두 줄로 적는다
                축2.text(가로, 세로, f"{칸_이름[세로][가로]}\n{값}건",
                         ha="center", va="center", color=글자색)
        그림2.colorbar(칠한것, ax=축2, label="건수")
        그림2.tight_layout()
        경로2 = 그림_폴더 / "02_혼동행렬.png"
        그림2.savefig(경로2, dpi=120)
        st.pyplot(그림2)
        plt.close(그림2)
        st.caption(f"시험용 {len(시험_정답)}행이 네 칸에 어떻게 나뉘었는지 - "
                   f"오른쪽 아래(잡은 것)가 진할수록 좋고, 왼쪽 아래(놓친 것)가 진할수록 아픕니다 "
                   f"(저장 : {경로2.name})")

        # --- 그림 3) 기준 모델과 내 모델의 점수를 나란히 놓은 막대그림 ---
        # 위에서 지금 문턱으로 다시 매긴 점수_표 를 그대로 쓴다
        지표들 = ["정확도", "정밀도", "재현율", "F1"]
        내_값 = [float(점수_표.iloc[0][지표]) for 지표 in 지표들]
        기준_값 = [float(점수_표.iloc[1][지표]) for 지표 in 지표들]
        자리 = list(range(len(지표들)))

        그림3, 축3 = plt.subplots(figsize=(7, 4.5))
        # 같은 지표를 왼쪽 오른쪽에 나란히 놓으려고 자리를 조금씩 밀어준다
        막대1 = 축3.bar([x - 0.2 for x in 자리], 내_값, width=0.4,
                      label=f"{모델_이름} (문턱 {문턱:.2f})", color="#4C78A8")
        막대2 = 축3.bar([x + 0.2 for x in 자리], 기준_값, width=0.4,
                      label="전부 정상이라 답하는 기준 모델", color="#BAB0AC")
        축3.bar_label(막대1, fmt="%.3f", fontsize=8)      # 막대 위에 숫자를 적는다
        축3.bar_label(막대2, fmt="%.3f", fontsize=8)
        축3.set_xticks(자리, 지표들)
        축3.set_ylim(0, 1.15)                             # 막대 위 숫자가 잘리지 않게 위를 넉넉히 둔다
        축3.set_ylabel("점수 (1에 가까울수록 좋음)")
        축3.set_title("기준 모델과 견준 점수")
        축3.legend(loc="upper right", fontsize=8)
        그림3.tight_layout()
        경로3 = 그림_폴더 / "03_점수비교.png"
        그림3.savefig(경로3, dpi=120)
        st.pyplot(그림3)
        plt.close(그림3)
        st.caption(f"네 가지 점수를 기준 모델과 나란히 놓은 것 - "
                   f"파란 막대가 회색 막대보다 높아야 배운 값어치가 있는 것입니다 (저장 : {경로3.name})")

# 탭 5 - 리포트
with 탭들[4]:
    # 1) 맨 위 - 프로젝트 요약 다섯 줄 (notes.md 에서 읽어온다)
    st.write("프로젝트 요약 다섯 줄")

    다섯줄 = 다섯줄_읽기()
    if not 다섯줄:
        # 파일에 그 대목이 없으면 지어내지 않고 어디를 봐야 하는지만 알려준다
        st.warning(f"{노트_경로.name} 에서 '리포트 뼈대 다섯 줄' 대목을 찾지 못했습니다")
    else:
        for 머리, 몸 in 다섯줄:
            st.markdown(f"**{머리}**")
            st.write(몸)
        st.caption(f"위 다섯 줄은 {노트_경로.name} 에서 그대로 읽어온 것입니다 "
                   f"- 파일을 고치면 이 화면도 같이 바뀝니다")

    st.write("---")

    if "리포트_자료" not in st.session_state:
        # 결과 탭을 아직 안 봤으면 문턱이 정해지지 않았으므로 그것만 알려준다
        st.write("'결과' 탭에서 문턱을 정하면 아래가 채워집니다")
    else:
        # 결과 탭이 넘겨준 꾸러미를 그대로 쓴다 - 여기서 다시 셈하지 않는다
        자료 = st.session_state["리포트_자료"]
        가중치_글 = "켬" if 자료["가중치_켬"] else "끔"
        st.write(f"'결과' 탭의 문턱 {자료['문턱']:.2f} 를 그대로 따라갑니다 "
                 f"- 모델 '{자료['모델_이름']}', 적은 쪽 가중치 {가중치_글}")

        # 2) 그 아래 - 결과 표 (기준 모델과 내 모델)
        st.write("결과 표")
        st.dataframe(자료["점수_표"], hide_index=True)

        # 3) 그 아래 - 지금 문턱에서의 건수 세 가지
        st.write(f"문턱 {자료['문턱']:.2f} 에서의 건수 "
                 f"(시험용 {자료['시험_행수']}행, 그중 진짜 불량 {자료['전체_불량']}건)")
        칸1, 칸2, 칸3 = st.columns(3)
        칸1.metric("지목한 건수", f"{자료['지목_건수']}건")
        칸2.metric("그중 진짜 불량", f"{자료['진짜_건수']}건")
        칸3.metric("놓친 불량", f"{자료['놓친_건수']}건")

    st.write("---")

    # 4) 맨 아래 - 해석 문장. 단추를 누를 때만 부른다
    st.write("해석 문장")

    if "리포트_자료" not in st.session_state:
        st.write("'결과' 탭에서 문턱을 정하면 해석 문장을 만들 수 있습니다")
    else:
        자료 = st.session_state["리포트_자료"]
        지금_도장 = 숫자_도장(자료)
        받은것 = st.session_state.get("해석_문장")

        # 열쇠를 .env 와 비밀 값 두 군데에서 찾는다
        # - 어디서 찾았는지만 알리고, 값은 화면에 절대 내보내지 않는다
        열쇠, 열쇠_자리 = 열쇠_찾기()

        if not 열쇠:
            # 두 자리 어디에도 없을 때 - 한 줄만 보여주고 여기서 멈추지 않는다
            st.warning(열쇠_없음_말)
        else:
            st.caption(f"열쇠는 {열쇠_자리} 에서 읽었습니다 (화면에는 보여주지 않습니다)")

            # 단추를 누를 때만 부른다 - 슬라이더를 옮기는 것만으로는 부르지 않는다
            if st.button("해석 문장 만들기"):
                if 받은것 and 받은것.get("도장") == 지금_도장:
                    # 같은 숫자로 이미 만든 문장이 있으면 다시 부르지 않는다
                    st.caption("같은 숫자로 이미 만든 문장이 있어 그대로 보여줍니다")
                else:
                    with st.spinner("문장을 받아오는 중입니다"):
                        문장, 알릴_말, _ = 해석_문장_받기(자료, 열쇠)
                    if 알릴_말:
                        # 빨간 글씨 대신 한국어 한 줄로만 알린다
                        # - 위쪽 요약과 표는 그대로 두고 여기서 멈추지 않는다
                        st.warning(알릴_말)
                    else:
                        # 어떤 숫자로 받은 문장인지 도장을 같이 찍어 둔다
                        st.session_state["해석_문장"] = {
                            "글": 문장, "문턱": 자료["문턱"], "도장": 지금_도장}
                        받은것 = st.session_state["해석_문장"]

        # 만들어 둔 문장이 있으면, 부르기에 실패했더라도 그대로 보여준다
        if 받은것:
            st.write(받은것["글"])
            if 받은것.get("도장") != 지금_도장:
                # 문장을 받은 뒤 손잡이를 옮겼으면 숫자가 어긋나므로 알려준다
                st.warning(f"이 문장은 문턱 {받은것['문턱']:.2f} 의 숫자로 받은 것입니다 "
                           f"- 지금 화면 숫자와 다릅니다. "
                           f"맞추려면 단추를 다시 눌러주세요")
            else:
                st.caption(f"문턱 {받은것['문턱']:.2f} 의 숫자로 받은 문장입니다")

    # 5) 맨 아래 - PDF 내려받기 단추
    st.write("---")
    st.write("PDF 로 내려받기")

    if "리포트_자료" not in st.session_state:
        st.write("'결과' 탭에서 문턱을 정하면 내려받을 수 있습니다")
    else:
        자료 = st.session_state["리포트_자료"]

        # 화면에 지금 떠 있는 것만 담는다 - 없는 값을 지어내지 않는다
        받은것 = st.session_state.get("해석_문장")
        # 문장을 받은 뒤 손잡이를 옮겼으면 숫자가 어긋나므로 그 문장은 넣지 않는다
        넣을_문장 = (받은것["글"] if 받은것 and 받은것.get("도장") == 숫자_도장(자료)
                 else None)

        오늘 = datetime.now().strftime("%Y-%m-%d")
        # 화면을 다시 그릴 때마다 지금 값으로 새로 만든다
        # - 그래서 손잡이를 옮긴 뒤 누르면 바뀐 숫자가 담긴다
        피디에프 = 리포트_피디에프_만들기(자료, 다섯줄_읽기(), 넣을_문장, 오늘)

        if 피디에프 is None:
            st.error("한글 글꼴을 찾지 못해 PDF 를 만들 수 없습니다")
        else:
            if 넣을_문장 is None:
                st.caption("지금 문턱의 해석 문장이 없어 그 자리는 "
                           "'아직 만들지 않았습니다' 로 들어갑니다")
            st.download_button(
                "PDF 내려받기",
                data=피디에프,                      # 서버에 저장하지 않고 바로 내려보낸다
                file_name=f"secom_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf")
            st.caption(f"담기는 문턱 : {자료['문턱']:.2f} "
                       f"(손잡이를 옮기고 다시 누르면 바뀐 숫자로 새로 만들어집니다)")

# 화면 맨 아래에 지금 시각을 보여준다 - 화면이 다시 그려질 때마다 새로 찍힌다
st.write("지금 시각 :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
