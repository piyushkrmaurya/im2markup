import os
import numpy as np
from tqdm import tqdm
from eval import generate_latex

test_images = [ "3279c959cb.png", "759aabf76d.png", "2d245d731b.png", "3cad98065b.png", "4493a352bf.png", "45669a7410.png", "26bc999853.png", "6f936fc6f8.png", "4a9b646eae.png", "cf8be841c0.png", "6f285970b2.png", "25cb952b0e.png", "167093e081.png", "125d4ab561.png", "ecddd11060.png", "23fd997703.png", "788cd2029a.png", "6fff29e23b.png", "1b027eec83.png", "36ec8d8219.png", "732db724ed.png", "1ec68dd740.png", "5d86173df3.png", "238b7b8ec9.png", "684ad21f24.png", "47062559f5.png", "58c383b1ae.png", "6eddbd7132.png", "7e98d733d8.png", "4e1d284923.png", "552bd92af0.png", "22bbb06ae0.png", "79230f17fe.png", "29cfc83c51.png", "401dcb5ed1.png", "242bb1dae9.png", "75fce7952d.png", "3f98ac87eb.png", "48f0408192.png", "7a4e32d2b7.png", "2580b33663.png", "5289c94dda.png", "65d002d775.png", "44d1a5f045.png", "aaf970fd61.png", "582c103a75.png", "5f3e72a4e5.png", "4c0c440893.png", "64f78ba1f5.png", "1e3140afa0.png", "39b9720b58.png", "6d688de275.png", "367e6af669.png", "164a0213cf.png", "28eb565379.png", "f9c2ae826f.png", "6a8ef3670a.png", "7f41e279e1.png", "2279dee9a4.png", "6705236376.png", "3633c994b3.png", "64958d56cf.png", "632bb44fb3.png", "589174dc5c.png", "65c2f46b49.png", "37d342638e.png", "6486e6da88.png", "4e7600585c.png", "47f637b516.png", "5169497afe.png", "13f0483499.png", "5143aa5a61.png", "767d513aab.png", "67aaf41e2c.png", "6b8690771f.png", "111e44a1c7.png", "5d12dfb239.png", "1716ec0f3a.png", "279b2873e3.png", "13985c7092.png", "479e0eed3b.png", "74fa2d5923.png", "35d3957b60.png", "368cce3588.png", "3e91211e02.png", "77b8f1fc09.png", "54487e387f.png", "68cc8a3725.png", "1d433c19b2.png", "1f2d4f35a1.png", "2dc38ee0da.png", "19e28d4629.png", "4393accf9d.png", "6d1205cf1c.png", "6ad4bf56aa.png", "3778e430ab.png", "440f4ec5b8.png", "3c4e681893.png", "457b458b5c.png", "7a09a75dd1.png", "46e5c1ecce.png", "c5f33441ce.png", "2423533128.png", "323fba3cb0.png", "30bcd541af.png", "4ab7aa393a.png", "189ae99f75.png", "761fcd9507.png", "2901cbfdb8.png", "618da453aa.png", "9ac2392176.png", "4d9c1045e0.png", "737f440710.png", "499bd4042a.png", "1050993b17.png", "e7f351e15a.png", "38da91b6e9.png", "3ccf41af9b.png", "b41c3c3917.png", "5a4f7b6933.png", "2e99ffa692.png", "2ea5d693dd.png", "1c098f5749.png", "5ebb96dfec.png", "4448e60e33.png", "20b4ddf1a7.png", "5bfffb9e71.png", "59d019d5bd.png", "26314ef7c7.png", "629c392f95.png", "cb15071605.png", "5c148631f6.png", "9543abec97.png", "a46a7fa19f.png", "40fd9df63b.png", "19a3b4f1bc.png", "5391988823.png", "46fbe38151.png", "ca67174d92.png", "683e077c75.png", "6f02640e60.png", "387e4595ec.png", "5ade99e8dd.png", "59341aee53.png", "3413b0ae51.png", "11ac9d2fdb.png", "de1eced60f.png", "381b171898.png", "76ca6c3c00.png", "3706294a74.png", "3395f77fc6.png", "74d053688c.png", "4d64719725.png", "18e2b0421a.png", "7314163c73.png", "46b174190d.png", "367f908aa2.png", "62952ff83c.png", "5a1fafa967.png", "66e2b7e912.png", "798dfc3825.png", "7983103370.png", "1576f9b769.png", "2808c824fc.png", "3e78c57c92.png", "56a2df656b.png", "36b093915a.png", "4d43d193c2.png", "17a8d1766f.png", "5016074f4b.png", "25f72f123c.png", "1255d2395a.png", "1cb56e01b7.png", "6edc5f3e4c.png", "4123cd5faf.png", "4dd5f76a64.png", "267583f14d.png", "51f2076b24.png", "17d2d1ea6c.png", "3a8c0abaac.png", "7512855fe8.png", "64676903ac.png", "374545d248.png", "27b2f4cb7b.png", "4a059e26e9.png", "33591d15dc.png", "1899c1ac76.png", "2135070bc4.png", "70cb3e946d.png", "4fdadb7d05.png", "668a392d2e.png", "2f3611e8d1.png"]

rows = ["<tr><th>Image</th><th>Rendered</th><th>Latex</th></tr>"]

for image in tqdm(test_images[:25]):
    image_path = "images/"+image
    latex = generate_latex(image_path)
    rows.append(f"<tr><td class='expression'><img src='{image_path}' /></td><td class='math'>${latex}$</td><td><pre>{latex.replace(' ', '')}</pre></td></tr>")

rows = "\n\t\t\t".join(rows)

html = f'''
<!Doctype HTML>
<html>
    <head>
         <style>
            body{{
                font-family: Arial
            }}
            table, td, th {{
                border: 1px solid #ddd;
                text-align: center;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
            }}

            td, th {{
                padding: 15px;
                width: 33%;
            }}
            pre {{
                white-space: pre-wrap; 
                white-space: -moz-pre-wrap;
                font-size: 12pt;
            }}
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/katex.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/katex.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/contrib/auto-render.min.js"></script>
    </head>
    <body>
        <table>
            {rows}
        </table>
        <script>
            renderMathInElement(
                document.body,
                {{
                    delimiters: [
                        {{ left: "$$", right: "$$", display: true }},
                        {{ left: "$", right: "$", display: false }},
                    ]
                }}
            );
        </script>
    </body>
</html>
'''

with open("index.html", "w") as f:
    f.write(html)