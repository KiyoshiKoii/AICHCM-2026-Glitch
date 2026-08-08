<#
.SYNOPSIS
    Gom data thô của BTC vào data/raw/ theo cấu trúc chuẩn của dự án.

.DESCRIPTION
    Nhận các thư mục BTC giao (Keyframes_L21, objects-aic25-b1, ...) đang nằm rải
    rác ở gốc repo, gom vào data/raw/ theo LOẠI dữ liệu thay vì theo đợt phát hành.
    Xem data/README.md để biết lý do.

    Script chỉ dùng Move-Item nên nếu nguồn và đích cùng ổ đĩa thì gần như tức thì,
    không tốn thêm dung lượng. Chạy lại nhiều lần được (idempotent): thư mục nào đã
    gom rồi thì bỏ qua.

    MẶC ĐỊNH LÀ DRY RUN — chỉ in ra định làm gì. Thêm -Execute để chạy thật.

.PARAMETER Source
    Nơi chứa các thư mục BTC. Mặc định là gốc repo.

.PARAMETER Execute
    Chạy thật. Không có cờ này thì chỉ in ra kế hoạch.

.PARAMETER SkipBusyCheck
    Bỏ qua bước cảnh báo "hình như còn đang copy dở".

.EXAMPLE
    powershell -File scripts/organize_data.ps1
    powershell -File scripts/organize_data.ps1 -Execute
#>
[CmdletBinding()]
param(
    [string]$Source,
    [switch]$Execute,
    [switch]$SkipBusyCheck
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = $RepoRoot }
$DataRoot = Join-Path $RepoRoot 'data'
$RawRoot = Join-Path $DataRoot 'raw'
$ExternalRoot = Join-Path $DataRoot 'external'

# Pattern thư mục BTC -> thư mục đích trong data/raw/.
# Pattern có wildcard để đợt b2, b3... sau này tự khớp, không phải sửa script.
$Mapping = [ordered]@{
    'Keyframes_L*'        = 'keyframes'
    'objects-aic*'        = 'objects'
    'clip-features-32*'   = 'clip-features-32'
    'map-keyframes-aic*'  = 'map-keyframes'
    'media-info-aic*'     = 'media-info'
}

# Thư mục không phải data BTC, chuyển nguyên khối sang data/external/.
$ExternalPatterns = @('vbs2023-*')

function Write-Step { param([string]$Message) Write-Host "  $Message" }
function Write-Skip { param([string]$Message) Write-Host "  - $Message" -ForegroundColor DarkGray }
function Write-Warn { param([string]$Message) Write-Host "  ! $Message" -ForegroundColor Yellow }

# Data BTC hay được đóng gói lồng thêm một lớp: Keyframes_L21/keyframes/L21_V001.
# Bóc các lớp bọc đó ra để lấy đúng thư mục chứa danh sách video.
function Resolve-PayloadDir {
    param([string]$Path)

    while ($true) {
        $children = @(Get-ChildItem -LiteralPath $Path -Force)
        $dirs = @($children | Where-Object { $_.PSIsContainer })
        $files = @($children | Where-Object { -not $_.PSIsContainer })

        # Chỉ bóc khi có đúng 1 thư mục con và không có file lạc.
        if ($dirs.Count -ne 1 -or $files.Count -gt 0) { break }
        # Đừng bóc nhầm khi thư mục chỉ chứa vỏn vẹn 1 video.
        if ($dirs[0].Name -match '^L\d+_V\d+$') { break }

        $Path = $dirs[0].FullName
    }
    return $Path
}

# Phát hiện transfer còn đang chạy.
#
# KHÔNG dùng LastWriteTime được: Windows giữ nguyên timestamp gốc khi *move* file,
# nên thư mục vừa được Explorer đổ đầy 5 giây trước vẫn mang ngày tháng từ hôm tải
# về. Cách duy nhất đáng tin là chụp hai lần rồi so sánh xem có gì thay đổi không.
#
# Chỉ đếm cấp 1 và thư mục con MỚI TẠO GẦN NHẤT (nơi Explorer đang ghi tới) để
# khỏi phải quét cả trăm nghìn file mỗi lần kiểm tra. Xếp theo CreationTime chứ
# không theo tên: Explorer không copy theo thứ tự bảng chữ cái, và copy giữa hai ổ
# đĩa khác nhau luôn tạo thư mục mới nên CreationTime là mốc đáng tin.
function Get-BusySnapshot {
    param([string]$Path)

    $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue)
    $newest = $children | Sort-Object CreationTime | Select-Object -Last 1
    $newestCount = 0
    if ($null -ne $newest -and $newest.PSIsContainer) {
        $newestCount = @(Get-ChildItem -LiteralPath $newest.FullName -Force -ErrorAction SilentlyContinue).Count
    }
    return "$($children.Count)|$($newest.Name)|$newestCount"
}

function Test-LooksBusy {
    param([string]$Path, [int]$WaitSeconds = 5)

    $before = Get-BusySnapshot -Path $Path
    Start-Sleep -Seconds $WaitSeconds
    return $before -ne (Get-BusySnapshot -Path $Path)
}

Write-Host ''
if ($Execute) {
    Write-Host 'organize_data.ps1 - CHAY THAT' -ForegroundColor Green
} else {
    Write-Host 'organize_data.ps1 - DRY RUN (them -Execute de chay that)' -ForegroundColor Cyan
}
Write-Host "  nguon: $Source"
Write-Host "  dich : $RawRoot"
Write-Host ''

if ($Execute) {
    foreach ($dir in @($RawRoot, $ExternalRoot,
                       (Join-Path $DataRoot 'processed\metadata'),
                       (Join-Path $DataRoot 'processed\objects_index'),
                       (Join-Path $DataRoot 'processed\embeddings'))) {
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
}

$moved = 0
$skipped = 0
$conflicts = 0
$busy = $false

$plan = @()
foreach ($pattern in $Mapping.Keys) {
    foreach ($dir in @(Get-ChildItem -LiteralPath $Source -Directory -Filter $pattern -ErrorAction SilentlyContinue)) {
        $plan += [pscustomobject]@{ Source = $dir; Target = Join-Path $RawRoot $Mapping[$pattern]; Merge = $true }
    }
}
foreach ($pattern in $ExternalPatterns) {
    foreach ($dir in @(Get-ChildItem -LiteralPath $Source -Directory -Filter $pattern -ErrorAction SilentlyContinue)) {
        $plan += [pscustomobject]@{ Source = $dir; Target = Join-Path $ExternalRoot $dir.Name; Merge = $false }
    }
}

if ($plan.Count -eq 0) {
    Write-Host 'Khong tim thay thu muc data BTC nao o nguon.' -ForegroundColor Yellow
    Write-Host 'Neu data da gom xong roi thi day la ket qua binh thuong.'
    Write-Host ''
    exit 0
}

foreach ($item in $plan) {
    $srcDir = $item.Source
    Write-Host "[$($srcDir.Name)] -> $($item.Target)" -ForegroundColor White

    if (-not $SkipBusyCheck -and (Test-LooksBusy -Path $srcDir.FullName)) {
        Write-Warn 'Noi dung van dang thay doi - transfer chua xong.'
        Write-Warn 'Doi Explorer chay xong roi chay lai (hoac -SkipBusyCheck neu chac chan).'
        $busy = $true
        Write-Host ''
        continue
    }

    # Chuyển nguyên khối (data/external/) - không bóc lớp, không merge.
    if (-not $item.Merge) {
        if (Test-Path -LiteralPath $item.Target) {
            Write-Skip 'da ton tai o dich, bo qua'
            $skipped++
        } elseif ($Execute) {
            Move-Item -LiteralPath $srcDir.FullName -Destination $item.Target
            Write-Step 'da chuyen ca thu muc'
            $moved++
        } else {
            Write-Step 'se chuyen ca thu muc'
            $moved++
        }
        Write-Host ''
        continue
    }

    $payload = Resolve-PayloadDir -Path $srcDir.FullName
    if ($payload -ne $srcDir.FullName) {
        $relative = $payload.Substring($srcDir.FullName.Length).TrimStart('\')
        Write-Step "boc lop bao: $relative"
    }

    if ($Execute -and -not (Test-Path -LiteralPath $item.Target)) {
        New-Item -ItemType Directory -Path $item.Target -Force | Out-Null
    }

    $children = @(Get-ChildItem -LiteralPath $payload -Force)
    $localMoved = 0
    $localSkipped = 0
    $provenance = @()

    foreach ($child in $children) {
        $dest = Join-Path $item.Target $child.Name
        if (Test-Path -LiteralPath $dest) {
            # Không ghi đè: bản gốc BTC quý hơn sự tiện lợi.
            Write-Skip "trung ten, giu nguyen ban cu: $($child.Name)"
            $localSkipped++
            $conflicts++
            continue
        }
        if ($Execute) {
            Move-Item -LiteralPath $child.FullName -Destination $dest
        }
        # Gom theo loại làm mất dấu đợt phát hành, nên ghi lại trước khi mất.
        $videoId = [System.IO.Path]::GetFileNameWithoutExtension($child.Name)
        if ($videoId -match '^L\d+_V\d+$') {
            $provenance += "$videoId,$($srcDir.Name)"
        }
        $localMoved++
    }

    if ($Execute -and $provenance.Count -gt 0) {
        $provFile = Join-Path $RawRoot '_provenance.csv'
        if (-not (Test-Path -LiteralPath $provFile)) {
            Set-Content -LiteralPath $provFile -Value 'video_id,source_dir' -Encoding utf8
        }
        Add-Content -LiteralPath $provFile -Value $provenance -Encoding utf8
    }

    $verb = if ($Execute) { 'da chuyen' } else { 'se chuyen' }
    Write-Step "$verb $localMoved muc (bo qua $localSkipped)"
    $moved += $localMoved
    $skipped += $localSkipped

    # Dọn vỏ rỗng còn lại sau khi đã bê hết ruột đi.
    if ($Execute) {
        $leftover = @(Get-ChildItem -LiteralPath $srcDir.FullName -Recurse -Force)
        if ($leftover.Count -eq 0) {
            Remove-Item -LiteralPath $srcDir.FullName -Recurse -Force
            Write-Step 'da xoa thu muc vo rong'
        } else {
            Write-Warn "con $($leftover.Count) muc sot lai trong $($srcDir.Name) - kiem tra tay truoc khi xoa"
        }
    }
    Write-Host ''
}

Write-Host ('-' * 60)
Write-Host "Tong: $moved muc chuyen, $skipped bo qua, $conflicts trung ten"
if ($conflicts -gt 0) {
    Write-Warn 'Co muc trung ten - script KHONG ghi de. Kiem tra tay xem ban nao dung.'
}
if ($busy) {
    Write-Warn 'Mot so thu muc bi bo qua vi nghi con dang copy. Chay lai sau khi xong.'
}
if (-not $Execute) {
    Write-Host ''
    Write-Host 'Day moi la DRY RUN. Chay that:' -ForegroundColor Cyan
    Write-Host '  powershell -File scripts/organize_data.ps1 -Execute'
} else {
    Write-Host ''
    Write-Host 'Buoc tiep theo: python scripts/build_manifest.py'
}
Write-Host ''
