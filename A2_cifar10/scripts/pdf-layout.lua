-- PDF-only pagination. The Markdown text and original figures remain unchanged.
function Figure(figure)
  local learning_curves = false
  figure:walk({Image = function(image)
    if image.src == "figures/learning_curves.png" then
      learning_curves = true
    end
  end})
  if not learning_curves then return nil end
  local groups = {}
  for group = 1, 3 do
    groups[#groups + 1] = pandoc.RawBlock("latex", "\\clearpage")
    groups[#groups + 1] = pandoc.RawBlock("latex", string.format([[
\begin{figure}[H]
\centering
\includegraphics[height=0.84\textheight,keepaspectratio]{tmp/pdfs/assets/learning-curves-%d.pdf}
\caption{训练曲线（第 %d/3 组）}
\end{figure}
]], group, group))
  end
  groups[#groups + 1] = pandoc.RawBlock("latex", "\\clearpage")
  return groups
end

function Header(header)
  local title = pandoc.utils.stringify(header.content)
  if title == "5. 正式实验结果" then
    return {pandoc.RawBlock("latex", "\\Needspace{18\\baselineskip}"), header}
  elseif title == "5.1 正则化消融" or title == "5.2 深度、残差与集成" then
    return {pandoc.RawBlock("latex", "\\Needspace{10\\baselineskip}"), header}
  end
end

function Pandoc(document)
  local blocks = {}
  local references = false
  for _, block in ipairs(document.blocks) do
    if block.t == "Header" and pandoc.utils.stringify(block.content) == "参考资料" then
      references = true
      blocks[#blocks + 1] = pandoc.RawBlock("latex",
        "\\Needspace{14\\baselineskip}\\begingroup\\small\\setlength{\\parskip}{2pt}")
    end
    blocks[#blocks + 1] = block
  end
  if references then
    blocks[#blocks + 1] = pandoc.RawBlock("latex", "\\endgroup")
  end
  document.blocks = blocks
  return document
end
